import numpy as np
from scipy.stats import truncnorm

def truncated_normal_n(mean, std, lower=-1.0, upper=1.0, 
                       max_iter=10000, rng = np.random.RandomState(None)):
    """
    Draws samples from a truncated normal distribution for each element of mean/std.

    Parameters
    ----------
    mean : array-like
        Vector of means.
    std : array-like
        Vector of standard deviations.
    lower, upper : float
        Truncation bounds.
    max_iter : int
        Maximum number of rejection-sampling iterations.

    Returns
    -------
    np.ndarray
        A vector of truncated normal samples.
    """
    mean = np.atleast_1d(mean).astype(float)
    std  = np.atleast_1d(std).astype(float)
    std[std < 1e-12] = 1e-12

    assert mean.shape == std.shape, "mean and std must have same shape"

    n = mean.shape[0]
    res = np.empty(n, dtype=float)
    done = np.zeros(n, dtype=bool)

    count = 0
    while not np.all(done) and count < max_iter:
        # Draw for those who are not yet filled
        m = mean[~done]
        s = std[~done]
        # Minimum security
        samples = rng.normal(m, s)
        keep = (samples >= lower) & (samples <= upper)

        # Write in unfilled positions
        idx = np.flatnonzero(~done)
        res[idx[keep]] = samples[keep]
        done[idx[keep]] = True
        count += 1

  # ---- FALLBACK SAME AS sample_alpha ----
    if not np.all(done):
        idx_fail = np.flatnonzero(~done)
        res[idx_fail] = np.random.uniform(lower, upper, size=len(idx_fail))

    return res


def truncated_normal(mean, std, lower=-1.0, upper=1.0, max_iter=10000):
    """
    Draws samples from a truncated normal distribution for each element of mean/std.

    Parameters
    ----------
    mean : array-like
        Vector of means.
    std : array-like
        Vector of standard deviations.
    lower, upper : float
        Truncation bounds.
    max_iter : int
        Maximum number of rejection-sampling iterations.

    Returns
    -------
    np.ndarray
        A vector of truncated normal samples.
    """
    mean = np.atleast_1d(mean).astype(float)
    std  = np.atleast_1d(std).astype(float)
    
    assert mean.shape == std.shape, "mean et std doivent avoir la même forme"

    n = mean.shape[0]
    res = np.empty(n, dtype=float)
    done = np.zeros(n, dtype=bool)

    count = 0
    while not np.all(done) and count < max_iter:
        # Draw for those who are not yet filled
        m = mean[~done]
        s = std[~done]
        # Minimum security
        s = np.maximum(s, 1e-12)
        samples = np.random.normal(m, s)
        keep = (samples >= lower) & (samples <= upper)

        # Write in unfilled positions
        idx = np.flatnonzero(~done)
        res[idx[keep]] = samples[keep]
        done[idx[keep]] = True
        count += 1

# ---- FALLBACK SAME AS sample_alpha ----
    if not np.all(done):
        idx_fail = np.flatnonzero(~done)
        res[idx_fail] = np.random.uniform(lower, upper, size=len(idx_fail))

    return res




def make_json_serializable(obj):
    
    """
    Recursively converts numpy objects into native Python types
    so they can be serialized to JSON.

    Useful when saving simulation results.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    else:
        return obj
