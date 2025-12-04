import numpy as np


def true_distance_midplane_cost_single_dist(X1, m1, m2):
    """
    Compute the midplane-based separation cost for points in one cluster
    relative to the midplane between two cluster centroids.

    Parameters
    ----------
    X1 : np.ndarray
        Points belonging to cluster i, shape (n_i, n_features).
    m1 : np.ndarray
        Centroid of cluster i, shape (n_features,).
    m2 : np.ndarray
        Centroid of cluster j, shape (n_features,).

    Returns
    -------
    float
        Sum of squared distances of projected points to the midplane.
        Returns np.inf if m1 and m2 coincide.
    """
    X1 = np.asarray(X1, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)

    d = m1 - m2
    D2 = np.dot(d, d)

    # Degenerate case: identical centroids
    if D2 == 0:
        return np.inf

    # Midpoint between centroids
    c = (m1 + m2) / 2.0

    # Unit direction between centroids
    d_unit = d / np.linalg.norm(d)

    # Squared distance of projections onto d_unit
    dist1_sq = np.sum(((X1 - c) @ d_unit) ** 2)

    return float(dist1_sq)


def cfq_score(X, labels_k, centers_k):
    """
    Compute a k-means clustering quality criterion based on the ratio between
    midplane-based cluster separation and within-cluster compactness.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    labels_k : np.ndarray
        Cluster assignments of shape (n_samples,).
    centers_k : np.ndarray
        Cluster centroids of shape (k, n_features).

    Returns
    -------
    float
        Modified variance-ratio-like score. Higher values indicate better
        separation relative to compactness. Returns np.inf if compactness is zero.
    """
    X = np.asarray(X, dtype=np.float64)
    labels_k = np.asarray(labels_k)
    centers_k = np.asarray(centers_k, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    if X.shape[0] != labels_k.shape[0]:
        raise ValueError("labels_k must have the same length as X.")
    if centers_k.ndim != 2:
        raise ValueError("centers_k must be a 2D array (k, n_features).")
    if X.shape[1] != centers_k.shape[1]:
        raise ValueError("Feature dimension mismatch between X and centers_k.")

    unique_labels = np.unique(labels_k)
    k = centers_k.shape[0]

    if k != len(unique_labels):
        raise ValueError(
            "Number of unique labels does not match number of cluster centers."
        )

    # ------------------------------------------------------------------
    # Within-cluster compactness
    # ------------------------------------------------------------------
    compactness = 0.0
    for j in range(k):
        Xj = X[labels_k == j]
        mj = centers_k[j]
        compactness += np.sum((Xj - mj) ** 2)

    # ------------------------------------------------------------------
    # Between-cluster midplane separation
    # ------------------------------------------------------------------
    min_sep_costs = []
    for i in range(k):
        Xi = X[labels_k == i]
        mi = centers_k[i]

        costs = []
        for j in range(k):
            if i == j:
                continue
            mj = centers_k[j]
            cost_ij = true_distance_midplane_cost_single_dist(Xi, mi, mj)
            costs.append(cost_ij)

        min_sep_costs.append(min(costs))

    total_sep = float(sum(min_sep_costs))

    # ------------------------------------------------------------------
    # Separation / compactness ratio
    # ------------------------------------------------------------------
    if compactness > 0.0:
        return total_sep / compactness
    else:
        return np.inf

