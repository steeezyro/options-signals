"""The online-learning layer: how the model actually gets better in real time.

Four components, each answering a different question:

  TimeVaryingRegression (Kalman)  -- what are my coefficients *now*?
  AdaptiveConformal (ACI)         -- how wrong am I, with what coverage?
  ProbabilityCalibrator           -- are my stated probabilities honest?
  DriftDetector (ADWIN / Page-Hinkley) -- has the regime changed?

Design note on Kalman vs. RLS-with-forgetting.  They are the same algorithm up
to how the covariance is inflated: RLS multiplies by 1/lambda, Kalman adds Q.
The multiplicative form winds up **unboundedly** in directions the data does
not excite -- and an options model routinely goes hours with no informative
observation.  The next informative sample then produces a huge destabilising
parameter jump.  The additive form converges to a finite steady state set by
Q and R.  For a live system, use the Kalman form.  This is not a stylistic
preference; it is the difference between a model that degrades gracefully over
a quiet afternoon and one that detonates on the first trade after lunch.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = [
    "TimeVaryingRegression", "AdaptiveConformal", "ProbabilityCalibrator",
    "PageHinkley", "ADWIN", "brier_decomposition", "expected_calibration_error",
]


# ----------------------------------------------------------------------------
# 1. time-varying coefficients
# ----------------------------------------------------------------------------

class TimeVaryingRegression:
    """Kalman filter over a slowly-drifting linear model.

        beta_t = mu + Phi (beta_{t-1} - mu) + eta_t,   eta ~ N(0, Q)
        y_t    = x_t' beta_t + e_t,                    e   ~ N(0, R)

    ``Phi < 1`` gives mean reversion to ``mu`` instead of a random walk.  Real
    financial coefficients do not wander to infinity, and a pure random walk
    prior lets them.

    The only genuinely free parameter is the signal-to-noise ratio q = Q/R:
    bigger q adapts faster and is noisier.  Set it by maximising the filter
    log-likelihood on history, or so the implied steady-state gain matches a
    target memory length.

    Every step logs the standardised innovation ``v_t/sqrt(S_t)``, which should
    be iid N(0,1).  If its running variance drifts above 1 the model is too
    rigid (raise q); Ljung-Box on it detects unmodelled structure.  That stream
    is also a free concept-drift detector -- feed it to :class:`ADWIN`.
    """

    def __init__(self, n_features: int, q: float = 1e-5, R: float = 1.0,
                 phi: float = 1.0, prior_var: float = 1.0, mu: np.ndarray | None = None):
        self.k = int(n_features)
        self.beta = np.zeros(self.k)
        self.P = np.eye(self.k) * float(prior_var)
        self.Q = np.eye(self.k) * float(q) * float(R)
        self.R = float(R)
        self.phi = float(phi)
        self.mu = np.zeros(self.k) if mu is None else np.asarray(mu, float)
        self.n = 0
        self.loglik = 0.0
        self.innovations: deque[float] = deque(maxlen=4096)

    def predict(self, x) -> float:
        return float(np.asarray(x, float) @ self.beta)

    def update(self, x, y: float, R: float | None = None) -> dict:
        x = np.asarray(x, float).reshape(-1)
        Rt = self.R if R is None else float(R)

        # ---- predict
        self.beta = self.mu + self.phi * (self.beta - self.mu)
        self.P = self.phi**2 * self.P + self.Q

        # ---- update
        v = float(y) - float(x @ self.beta)                 # innovation
        S = float(x @ self.P @ x) + Rt                      # innovation variance
        K = (self.P @ x) / S
        self.beta = self.beta + K * v
        I_Kx = np.eye(self.k) - np.outer(K, x)
        # Joseph form: preserves symmetry and positive-definiteness under
        # floating point.  The textbook (I-Kx)P form loses PSD after a few
        # thousand updates and the filter quietly stops converging.
        self.P = I_Kx @ self.P @ I_Kx.T + np.outer(K, K) * Rt

        self.n += 1
        self.loglik += -0.5 * (np.log(2.0 * np.pi * S) + v * v / S)
        z = v / np.sqrt(S)
        self.innovations.append(float(z))
        return {"innovation": v, "S": S, "z": float(z), "beta": self.beta.copy()}

    def innovation_health(self) -> dict:
        z = np.asarray(self.innovations, float)
        if z.size < 30:
            return {"n": int(z.size), "var": np.nan, "mean": np.nan, "verdict": "warming-up"}
        var = float(np.var(z))
        verdict = "ok" if 0.6 <= var <= 1.6 else ("too-rigid (raise q)" if var > 1.6 else "too-loose (lower q)")
        return {"n": int(z.size), "var": var, "mean": float(np.mean(z)), "verdict": verdict}


# ----------------------------------------------------------------------------
# 2. adaptive conformal inference
# ----------------------------------------------------------------------------

class AdaptiveConformal:
    """Gibbs & Candes (2021) adaptive conformal inference.

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    Maintains long-run marginal coverage under *arbitrary* distribution shift,
    with no exchangeability assumption -- the right uncertainty framework for a
    live trading model, because market data is emphatically not exchangeable.

    The guarantee is deterministic and assumption-free::

        | (1/T) sum err_t - alpha |  <=  (max(alpha_1, 1-alpha_1) + gamma) / (gamma T)

    Interpretation for this system: the interval the sizer consumes is honest
    *by construction over time*, even across a regime break.  A model whose
    stated 90% interval actually covers 60% will overbet by a factor that no
    amount of Kelly fractioning repairs.

    A sustained run of ``alpha_t <= 0`` means the model has broken badly enough
    that only an infinite-width interval achieves coverage.  Treat it as a
    KILL SWITCH, not as a wide interval.
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.01, window: int = 500,
                 kill_after: int = 20):
        self.alpha_target = float(alpha)
        self.alpha_t = float(alpha)
        self.gamma = float(gamma)
        self.scores: deque[float] = deque(maxlen=int(window))
        self.errs: deque[int] = deque(maxlen=int(window))
        self._degenerate_run = 0
        self.kill_after = int(kill_after)

    def quantile(self) -> float:
        """Q_hat(1 - alpha_t) over the calibration scores."""
        if not self.scores:
            return np.inf
        lvl = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        return float(np.quantile(np.asarray(self.scores, float), lvl, method="higher"))

    def interval(self, point: float) -> tuple[float, float]:
        q = self.quantile()
        if not np.isfinite(q):
            return (-np.inf, np.inf)
        return (point - q, point + q)

    def update(self, point: float, actual: float) -> dict:
        score = abs(float(point) - float(actual))
        lo, hi = self.interval(point)
        err = 0 if (lo <= actual <= hi) else 1
        self.scores.append(score)
        self.errs.append(err)
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha_target - err), 0.0, 1.0))
        self._degenerate_run = self._degenerate_run + 1 if self.alpha_t <= 1e-12 else 0
        return {
            "score": score, "err": err, "alpha_t": self.alpha_t,
            "empirical_coverage": 1.0 - float(np.mean(self.errs)) if self.errs else np.nan,
            "killed": self._degenerate_run >= self.kill_after,
        }

    @property
    def killed(self) -> bool:
        return self._degenerate_run >= self.kill_after


# ----------------------------------------------------------------------------
# 3. probability calibration + scoring
# ----------------------------------------------------------------------------

class ProbabilityCalibrator:
    """Platt scaling below ~1000 calibration points, isotonic above.

    Uses Lin-Weng-Keerthi target smoothing (t+ = (N+ + 1)/(N+ + 2),
    t- = 1/(N- + 2)) which prevents the degenerate ``A -> -inf`` fit on
    separable data.

    Why this module exists at all: Kelly is a function of the probability
    *level*, not of the ranking.  A model can have excellent AUC and be badly
    miscalibrated, and sizing off an uncalibrated probability is the fastest
    route to systematic overbetting.  Reliability (the REL term of the Brier
    decomposition) is the metric that matters here, not accuracy.
    """

    def __init__(self, min_isotonic: int = 1000):
        self.min_isotonic = int(min_isotonic)
        self.mode_: str | None = None
        self._A, self._B = 0.0, 0.0
        self._iso = None

    def fit(self, scores, labels) -> "ProbabilityCalibrator":
        f = np.asarray(scores, float).ravel()
        y = np.asarray(labels, float).ravel()
        m = np.isfinite(f) & np.isfinite(y)
        f, y = f[m], y[m]
        if f.size == 0:
            return self
        if f.size >= self.min_isotonic:
            from sklearn.isotonic import IsotonicRegression  # optional dependency
            self._iso = IsotonicRegression(out_of_bounds="clip").fit(f, y)
            self.mode_ = "isotonic"
            return self

        n_pos, n_neg = float((y > 0.5).sum()), float((y <= 0.5).sum())
        t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        from scipy.optimize import minimize
        def nll(p):
            A, B = p
            z = np.clip(A * f + B, -35, 35)
            q = 1.0 / (1.0 + np.exp(z))
            q = np.clip(q, 1e-12, 1 - 1e-12)
            return float(-np.sum(t * np.log(q) + (1 - t) * np.log(1 - q)))
        res = minimize(nll, np.array([-1.0, 0.0]), method="Nelder-Mead")
        self._A, self._B = map(float, res.x)
        self.mode_ = "platt"
        return self

    def transform(self, scores):
        f = np.asarray(scores, float)
        if self.mode_ == "isotonic":
            return np.clip(self._iso.predict(f), 1e-6, 1 - 1e-6)
        if self.mode_ == "platt":
            return np.clip(1.0 / (1.0 + np.exp(np.clip(self._A * f + self._B, -35, 35))), 1e-6, 1 - 1e-6)
        return np.clip(f, 1e-6, 1 - 1e-6)


def brier_decomposition(p, y, n_bins: int = 15) -> dict:
    """Murphy decomposition  BS = REL - RES + UNC.

    REL (reliability, lower better) is the squared vertical distance from the
    diagonal on a reliability diagram.  It is *the* calibration metric.
    RES (resolution, higher better) is discrimination.
    """
    p = np.asarray(p, float).ravel(); y = np.asarray(y, float).ravel()
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    N = p.size
    if N == 0:
        return {"brier": np.nan, "rel": np.nan, "res": np.nan, "unc": np.nan, "n": 0}
    o_bar = float(y.mean())
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rel = res = 0.0
    for b in range(n_bins):
        s = idx == b
        nk = int(s.sum())
        if nk == 0:
            continue
        pk, ok = float(p[s].mean()), float(y[s].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - o_bar) ** 2
    rel /= N; res /= N
    return {"brier": float(np.mean((p - y) ** 2)), "rel": rel, "res": res,
            "unc": o_bar * (1 - o_bar), "n": int(N)}


def expected_calibration_error(p, y, n_bins: int = 15) -> float:
    p = np.asarray(p, float).ravel(); y = np.asarray(y, float).ravel()
    m = np.isfinite(p) & np.isfinite(y); p, y = p[m], y[m]
    if p.size == 0:
        return np.nan
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    return float(sum(
        (idx == b).sum() / p.size * abs(p[idx == b].mean() - y[idx == b].mean())
        for b in range(n_bins) if (idx == b).sum() > 0
    ))


# ----------------------------------------------------------------------------
# 4. drift detection
# ----------------------------------------------------------------------------

class PageHinkley:
    """One-sided CUSUM for a mean shift.  Run two for two-sided.

    Use on specific monitored scalars where you know the shift size you care
    about: realised-minus-implied spread, fill ratio, slippage per contract.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0,
                 alpha: float = 0.9999, min_instances: int = 30):
        self.delta, self.threshold = float(delta), float(threshold)
        self.alpha, self.min_instances = float(alpha), int(min_instances)
        self.reset()

    def reset(self):
        self.n = 0; self.x_mean = 0.0; self.m = 0.0; self.M = 0.0; self.drift = False

    def update(self, x: float) -> bool:
        x = float(x); self.n += 1
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * x if self.n > 1 else x
        self.m += x - self.x_mean - self.delta
        self.M = min(self.M, self.m)
        self.drift = (self.n >= self.min_instances) and ((self.m - self.M) > self.threshold)
        if self.drift:
            self.reset()
            return True
        return False


class ADWIN:
    """Adaptive windowing (Bifet & Gavalda).  Exact O(n^2)-split variant.

    Maintains a variable-length window; whenever any split shows sub-window
    means differing by more than the Hoeffding/Bernstein bound, the older part
    is dropped.  Two properties make it the right default on the *loss stream*:
    the false-positive rate is bounded by delta with no tuning, and the
    surviving window length is itself the correct adaptive lookback for
    refitting.

    This implementation stores raw values (bounded by ``max_len``) rather than
    the exponential-histogram buckets of the original.  At the sample rates
    here -- one observation per decision, not per tick -- that is fine and the
    code is auditable, which matters more.
    """

    def __init__(self, delta: float = 0.002, max_len: int = 2000, min_sub: int = 5):
        self.delta, self.max_len, self.min_sub = float(delta), int(max_len), int(min_sub)
        self.window: deque[float] = deque(maxlen=self.max_len)
        self.n_detections = 0

    @property
    def width(self) -> int:
        return len(self.window)

    @property
    def mean(self) -> float:
        return float(np.mean(self.window)) if self.window else np.nan

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        n = len(self.window)
        if n < 2 * self.min_sub:
            return False
        arr = np.asarray(self.window, float)
        csum = np.cumsum(arr)
        total, var = csum[-1], float(np.var(arr))
        dprime = self.delta / n
        detected = False
        for i in range(self.min_sub, n - self.min_sub + 1):
            n0, n1 = i, n - i
            mu0, mu1 = csum[i - 1] / n0, (total - csum[i - 1]) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = np.sqrt(2.0 / m * var * np.log(2.0 / dprime)) + 2.0 / (3.0 * m) * np.log(2.0 / dprime)
            if abs(mu0 - mu1) > eps:
                for _ in range(i):
                    self.window.popleft()
                detected = True
                self.n_detections += 1
                break
        return detected
