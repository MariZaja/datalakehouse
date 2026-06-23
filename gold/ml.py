import logging
import numpy as np
from typing import Dict, List, Optional
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class QualityAwareGaussianHMM(GaussianHMM):
    def __init__(
        self,
        n_states: int,
        modality_dims: Dict[str, int],
        state_names: Optional[List[str]] = None,
        transition_smoothing: float = 1.0,
        prior_smoothing: float = 1.0,
        var_floor: float = 1e-4,
        noise_strength: float = 20.0,
    ):
        super().__init__(n_components=n_states, covariance_type="full")
        self.modality_dims = dict(modality_dims)
        self.modalities = list(modality_dims.keys())
        self.state_names = state_names or [str(i) for i in range(n_states)]
        self.transition_smoothing = transition_smoothing
        self.prior_smoothing = prior_smoothing
        self.var_floor = var_floor
        self.noise_strength = noise_strength

        # Built during fit(); keyed by modality name
        self._vars_by_modality: Dict[str, np.ndarray] = {}   # (K, D, D) full covariance
        self._noise_vars_by_modality: Dict[str, float] = {}  # scalar noise per modality

        # Slice into the concatenated feature vector for each modality
        self._modality_slices: Dict[str, slice] = {}
        offset = 0
        for m, dim in modality_dims.items():
            self._modality_slices[m] = slice(offset, offset + dim)
            offset += dim

        # Set before decode/predict; cleared in finally — quality-aware emission hook
        self._current_quality: Optional[Dict[str, np.ndarray]] = None

    @property
    def n_states(self) -> int:
        return self.n_components

    def fit(self, sequences: List[Dict]) -> "QualityAwareGaussianHMM":
        K = self.n_components
        prior_counts = np.full(K, self.prior_smoothing)
        trans_counts = np.full((K, K), self.transition_smoothing)

        sum_w:    Dict[str, np.ndarray] = {}
        sum_wx:   Dict[str, np.ndarray] = {}
        sum_wxxT: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            dim = self.modality_dims[m]
            sum_w[m]    = np.zeros(K)
            sum_wx[m]   = np.zeros((K, dim))
            sum_wxxT[m] = np.zeros((K, dim, dim))

        for seq in sequences:
            y = np.asarray(seq["y"], dtype=int)
            self._validate_labels(y)
            prior_counts[y[0]] += 1.0
            for t in range(1, len(y)):
                trans_counts[y[t - 1], y[t]] += 1.0

            for m in self.modalities:
                obs = np.asarray(seq[f"obs_{m}"], dtype=float)
                q   = np.asarray(seq[f"q_{m}"],   dtype=float)
                T   = obs.shape[0]
                if q.shape != (T,):
                    raise ValueError(f"q_{m} shape mismatch: expected ({T},), got {q.shape}")
                q = np.clip(q, 0.01, 1.0)
                for state in range(K):
                    mask = y[:T] == state
                    if not np.any(mask):
                        continue
                    w_s = q[mask]
                    x_s = obs[mask]
                    sum_w[m][state]    += w_s.sum()
                    sum_wx[m][state]   += (w_s[:, None] * x_s).sum(axis=0)
                    sum_wxxT[m][state] += (w_s[:, None] * x_s).T @ x_s

        self.startprob_ = prior_counts / prior_counts.sum()
        self.transmat_  = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        total_dim = sum(self.modality_dims.values())
        self.n_features = total_dim  # required by GaussianHMM internals
        all_means  = np.zeros((K, total_dim))
        all_covars = np.zeros((K, total_dim, total_dim))

        for m in self.modalities:
            sl  = self._modality_slices[m]
            dim = self.modality_dims[m]
            m_means = np.zeros((K, dim))
            m_covs  = np.zeros((K, dim, dim))

            for state in range(K):
                sw = sum_w[m][state]
                if sw == 0.0:
                    raise ValueError(f"No training samples for state {state}, modality '{m}'")
                mean = sum_wx[m][state] / sw
                cov  = sum_wxxT[m][state] / sw - np.outer(mean, mean)
                cov  = 0.5 * (cov + cov.T)          # symmetrise numerical drift
                cov += self.var_floor * np.eye(dim)
                m_means[state] = mean
                m_covs[state]  = cov

            all_means[:, sl] = m_means
            for k in range(K):
                all_covars[k, sl.start:sl.stop, sl.start:sl.stop] = m_covs[k]

            self._vars_by_modality[m] = m_covs

            diag_vals = np.array([np.diag(m_covs[k]) for k in range(K)])  # (K, D)
            noise_scalar = float(np.maximum(
                self.noise_strength * np.median(diag_vals),
                self.var_floor,
            ))
            self._noise_vars_by_modality[m] = noise_scalar

            if self.noise_strength > 0.0:
                med_var = float(np.median(diag_vals))
                logger.info(
                    "Modality '%s': noise_strength=%.1f | median diag_var=%.4f"
                    " | noise_scalar=%.4f → variance inflation at q=0: +%.0f%%, at q=0.5: +%.0f%%",
                    m, self.noise_strength, med_var, noise_scalar,
                    100.0 * noise_scalar / max(med_var, 1e-10),
                    50.0  * noise_scalar / max(med_var, 1e-10),
                )

        self.means_  = all_means
        self.covars_ = all_covars
        return self

    def predict(
        self,
        obs: Dict[str, np.ndarray],
        q: Dict[str, np.ndarray],
        return_state_names: bool = True,
    ):
        obs, q = self._cast(obs, q)
        X = self._concat_modalities(obs)
        self._current_quality = q
        try:
            path = super().predict(X)
        finally:
            self._current_quality = None
        if return_state_names:
            return [self.state_names[i] for i in path]
        return path

    def predict_log_proba(
        self,
        obs: Dict[str, np.ndarray],
        q: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Smoothed log-posterior P(state_t | all obs). Shape: (T, n_states)."""
        obs, q = self._cast(obs, q)
        X = self._concat_modalities(obs)
        self._current_quality = q
        try:
            log_proba = np.log(super().predict_proba(X) + 1e-300)
        finally:
            self._current_quality = None
        return log_proba

    # ── hmmlearn hook ─────────────────────────────────────────────────────────

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Quality-inflated full Gaussian log-likelihood. Shape: (T, n_components)."""
        T = X.shape[0]
        K = self.n_components
        log_l = np.zeros((T, K))

        for m in self.modalities:
            sl      = self._modality_slices[m]
            X_m     = X[:, sl]                           # (T, D)
            D       = X_m.shape[1]
            covs_m  = self._vars_by_modality[m]           # (K, D, D)
            means_m = self.means_[:, sl]                  # (K, D)
            noise_s = self._noise_vars_by_modality[m]     # scalar

            q_m = (
                np.clip(self._current_quality[m], 0.0, 1.0)
                if self._current_quality is not None
                else np.ones(T)
            )  # (T,)

            eye_D = np.eye(D)

            # eff_cov[t, k] = covs_m[k] + (noise_s*(1-q_t) + var_floor) * I
            noise_per_t = noise_s * (1.0 - q_m) + self.var_floor          # (T,)
            eff_cov = (
                covs_m[None, :, :, :]                                           # (1, K, D, D)
                + noise_per_t[:, None, None, None] * eye_D[None, None, :, :]   # (T, 1, D, D)
            )  # (T, K, D, D)

            _, log_det = np.linalg.slogdet(eff_cov)                            # (T, K)
            inv_cov    = np.linalg.inv(eff_cov)                                # (T, K, D, D)

            diff  = X_m[:, None, :] - means_m[None, :, :]                     # (T, K, D)
            mahal = np.einsum("tkd,tkde,tke->tk", diff, inv_cov, diff)         # (T, K)

            log_l += -0.5 * (D * np.log(2.0 * np.pi) + log_det + mahal)

        return log_l

    # ── helpers ───────────────────────────────────────────────────────────────

    def _concat_modalities(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs[m] for m in self.modalities], axis=1)

    def _validate_labels(self, y: np.ndarray) -> None:
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if np.any(y < 0) or np.any(y >= self.n_components):
            raise ValueError(f"y contains labels outside [0, {self.n_components - 1}]")

    def _cast(self, obs, q):
        return (
            {m: np.asarray(v, dtype=float) for m, v in obs.items()},
            {m: np.asarray(v, dtype=float) for m, v in q.items()},
        )
