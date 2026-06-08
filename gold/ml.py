import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityAwareGaussianHMM:
    """
    Quality-aware Gaussian HMM with N independent diagonal-Gaussian emission streams.

    Each modality m contributes:
        obs_m_t | state, q_m_t ~ N(mean_m[state], var_m[state] + noise_m*(1 - q_m_t))

    Log-likelihoods are summed across modalities (conditional independence given state).

    noise_strength controls how quality influences inference:
        0.0  → quality only affects training (weighted mean/var estimation); inference uses
               fixed per-state variance regardless of window quality.
        >0.0 → quality also inflates emission variance during inference — low-quality windows
               (q→0) increase variance by noise_strength × median_var, making the model less
               confident about poorly-acquired observations.

    Calibration grid to explore: 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0

    fit() expects sequences with keys:
        obs_<modality>: np.ndarray (T, dim)
        q_<modality>:   np.ndarray (T,)   quality in [0, 1], higher = better
        y:              np.ndarray (T,)   integer labels 0..n_states-1

    predict() / predict_log_proba() expect dicts keyed by modality name.
    """

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
        self.n_states = n_states
        self.modality_dims = dict(modality_dims)
        self.modalities = list(modality_dims.keys())
        self.state_names = state_names or [str(i) for i in range(n_states)]
        self.transition_smoothing = transition_smoothing
        self.prior_smoothing = prior_smoothing
        self.var_floor = var_floor
        self.noise_strength = noise_strength

        self.log_prior_: Optional[np.ndarray] = None
        self.log_transition_: Optional[np.ndarray] = None
        self.means_: Dict[str, np.ndarray] = {}
        self.vars_: Dict[str, np.ndarray] = {}
        self.noise_vars_: Dict[str, np.ndarray] = {}

    def fit(self, sequences: List[Dict]) -> "QualityAwareGaussianHMM":
        prior_counts = np.full(self.n_states, self.prior_smoothing)
        trans_counts = np.full((self.n_states, self.n_states), self.transition_smoothing)

        # Incremental weighted sufficient statistics: sum(w), sum(w*x), sum(w*x^2).
        # Avoids accumulating all training arrays in memory simultaneously.
        sum_w:  Dict[str, np.ndarray] = {}
        sum_wx: Dict[str, np.ndarray] = {}
        sum_wx2: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            dim = self.modality_dims[m]
            sum_w[m]   = np.zeros((self.n_states,))
            sum_wx[m]  = np.zeros((self.n_states, dim))
            sum_wx2[m] = np.zeros((self.n_states, dim))

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
                for state in range(self.n_states):
                    mask = y[:T] == state
                    if not np.any(mask):
                        continue
                    w_s = q[mask]
                    x_s = obs[mask]
                    sum_w[m][state]   += w_s.sum()
                    sum_wx[m][state]  += (w_s[:, None] * x_s).sum(axis=0)
                    sum_wx2[m][state] += (w_s[:, None] * x_s ** 2).sum(axis=0)

        self.log_prior_ = np.log(prior_counts / prior_counts.sum())
        trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        self.log_transition_ = np.log(trans_probs)

        for m in self.modalities:
            dim = self.modality_dims[m]
            means = np.zeros((self.n_states, dim))
            vars_ = np.zeros((self.n_states, dim))

            for state in range(self.n_states):
                sw = sum_w[m][state]
                if sw == 0.0:
                    raise ValueError(f"No training samples for state {state}, modality '{m}'")
                mean = sum_wx[m][state] / sw
                var  = np.maximum(sum_wx2[m][state] / sw - mean ** 2, self.var_floor)
                means[state] = mean
                vars_[state] = var

            self.means_[m] = means
            self.vars_[m] = vars_
            # Noise added when quality is low — scales with per-feature variance magnitude.
            self.noise_vars_[m] = np.maximum(
                self.noise_strength * np.median(vars_, axis=0),
                self.var_floor,
            )

            if self.noise_strength > 0.0:
                med_var = float(np.median(vars_))
                med_noise = float(np.median(self.noise_vars_[m]))
                logger.info(
                    "Modality '%s': noise_strength=%.1f | median state_var=%.4f"
                    " | median noise_var=%.4f → variance inflation at q=0: +%.0f%%, at q=0.5: +%.0f%%",
                    m, self.noise_strength, med_var, med_noise,
                    100.0 * med_noise / max(med_var, 1e-10),
                    50.0 * med_noise / max(med_var, 1e-10),
                )

        return self

    def predict(
        self,
        obs: Dict[str, np.ndarray],
        q: Dict[str, np.ndarray],
        return_state_names: bool = True,
    ):
        self._check_is_fitted()
        obs, q = self._cast(obs, q)
        path = self._viterbi(self._emission_log_probs(obs, q))
        if return_state_names:
            return [self.state_names[i] for i in path]
        return path

    def predict_log_proba(
        self,
        obs: Dict[str, np.ndarray],
        q: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Filtered log-posterior P(state_t | obs_1..t). Shape: (T, n_states)."""
        self._check_is_fitted()
        obs, q = self._cast(obs, q)
        return self._forward(self._emission_log_probs(obs, q))

    # ── internals ──────────────────────────────────────────────────────────────

    def _emission_log_probs(
        self,
        obs: Dict[str, np.ndarray],
        q: Dict[str, np.ndarray],
    ) -> np.ndarray:
        T = next(iter(obs.values())).shape[0]
        elp = np.zeros((T, self.n_states))

        for m in self.modalities:
            obs_m = obs[m]
            q_m = np.clip(q[m], 0.0, 1.0)
            noise = self.noise_vars_[m]
            for t in range(T):
                for state in range(self.n_states):
                    var_t = self.vars_[m][state] + noise * (1.0 - q_m[t]) + self.var_floor
                    elp[t, state] += self._diag_gaussian_logpdf(
                        obs_m[t], self.means_[m][state], var_t
                    )

        return elp

    def _viterbi(self, elp: np.ndarray) -> np.ndarray:
        T, K = elp.shape
        dp = np.empty((T, K))
        bp = np.zeros((T, K), dtype=int)
        dp[0] = self.log_prior_ + elp[0]
        for t in range(1, T):
            for s in range(K):
                cands = dp[t - 1] + self.log_transition_[:, s]
                best = int(np.argmax(cands))
                dp[t, s] = cands[best] + elp[t, s]
                bp[t, s] = best
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        return path

    def _forward(self, elp: np.ndarray) -> np.ndarray:
        T, K = elp.shape
        alpha = np.empty((T, K))
        alpha[0] = self.log_prior_ + elp[0]
        alpha[0] -= self._logsumexp(alpha[0])
        for t in range(1, T):
            for s in range(K):
                alpha[t, s] = elp[t, s] + self._logsumexp(alpha[t - 1] + self.log_transition_[:, s])
            alpha[t] -= self._logsumexp(alpha[t])
        return alpha

    def _weighted_mean_var(self, x: np.ndarray, w: np.ndarray):
        w = np.clip(w, 0.01, 1.0)
        w = w / w.sum()
        mean = np.sum(x * w[:, None], axis=0)
        var = np.maximum(np.sum(w[:, None] * (x - mean) ** 2, axis=0), self.var_floor)
        return mean, var

    def _validate_labels(self, y: np.ndarray) -> None:
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if np.any(y < 0) or np.any(y >= self.n_states):
            raise ValueError(f"y contains labels outside [0, {self.n_states - 1}]")

    def _check_is_fitted(self) -> None:
        if self.log_prior_ is None:
            raise RuntimeError("Model not fitted. Call fit(sequences) first.")

    def _cast(self, obs, q):
        return (
            {m: np.asarray(v, dtype=float) for m, v in obs.items()},
            {m: np.asarray(v, dtype=float) for m, v in q.items()},
        )

    @staticmethod
    def _diag_gaussian_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        return float(-0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum((x - mean) ** 2 / var)))

    @staticmethod
    def _logsumexp(a: np.ndarray) -> float:
        a_max = np.max(a)
        return float(a_max + np.log(np.sum(np.exp(a - a_max))))
