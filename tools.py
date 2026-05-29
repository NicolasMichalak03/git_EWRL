import numpy as np
from scipy.stats import truncnorm

def truncated_normal_n(mean, std, lower=-1.0, upper=1.0, 
                       max_iter=10000, rng = np.random.RandomState(None)):
    """
    Tire une loi normale tronquée pour chaque élément de mean/std.
    - mean, std: vecteurs de même forme (n,)
    - Retour: un vecteur (n,) avec un échantillon tronqué par élément.

    This is used because a lot faster than scipy.
    """
    mean = np.atleast_1d(mean).astype(float)
    std  = np.atleast_1d(std).astype(float)
    std[std < 1e-12] = 1e-12

    assert mean.shape == std.shape, "mean et std doivent avoir la même forme"

    n = mean.shape[0]
    res = np.empty(n, dtype=float)
    done = np.zeros(n, dtype=bool)

    count = 0
    while not np.all(done) and count < max_iter:
        # Tirage pour ceux qui ne sont pas encore remplis
        m = mean[~done]
        s = std[~done]
         # Sécurité minimale
        samples = rng.normal(m, s)
        keep = (samples >= lower) & (samples <= upper)

        # Écrire dans les positions non remplies
        idx = np.flatnonzero(~done)
        res[idx[keep]] = samples[keep]
        done[idx[keep]] = True
        count += 1

  # ---- FALLBACK IDENTIQUE À sample_alpha ----
    if not np.all(done):
        idx_fail = np.flatnonzero(~done)
        res[idx_fail] = np.random.uniform(lower, upper, size=len(idx_fail))

    return res


def truncated_normal(mean, std, lower=-1.0, upper=1.0, max_iter=10000, rng = None):
    """
    Tire une loi normale tronquée pour chaque élément de mean/std.
    - mean, std: vecteurs de même forme (n,)
    - Retour: un vecteur (n,) avec un échantillon tronqué par élément.
    """
    mean = np.atleast_1d(mean).astype(float)
    std  = np.atleast_1d(std).astype(float)
    assert mean.shape == std.shape, "mean et std doivent avoir la même forme"

    n = mean.shape[0]
    res = np.empty(n, dtype=float)
    done = np.zeros(n, dtype=bool)

    count = 0
    while not np.all(done) and count < max_iter:
        # Tirage pour ceux qui ne sont pas encore remplis
        m = mean[~done]
        s = std[~done]
         # Sécurité minimale
        s = np.maximum(s, 1e-12)
        samples = np.random.normal(m, s)
        keep = (samples >= lower) & (samples <= upper)

        # Écrire dans les positions non remplies
        idx = np.flatnonzero(~done)
        res[idx[keep]] = samples[keep]
        done[idx[keep]] = True
        count += 1

  # ---- FALLBACK IDENTIQUE À sample_alpha ----
    if not np.all(done):
        idx_fail = np.flatnonzero(~done)
        res[idx_fail] = np.random.uniform(lower, upper, size=len(idx_fail))

    return res



def make_json_serializable(obj):
    """Convertit récursivement les objets numpy en types Python natifs."""
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
