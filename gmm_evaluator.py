import numpy as np
from scipy.linalg import inv
from scipy.optimize import root


def compute_actionable_counterfactual(m1, S1, m2, S2, y, M):
    """
    Compute actionable counterfactuals for Gaussian clusters with full covariance.

    Args:
    - m1: Mean vector for p1(x).
    - S1: Covariance matrix for p1(x).
    - m2: Mean vector for p2(x).
    - S2: Covariance matrix for p2(x).
    - y: Factual vector.
    - M: Mask vector (1 for free indices, 0 for fixed indices).

    Returns:
    - z: Counterfactual vector.
    """
    # Partition indices
    F = np.where(M == 1)[0]
    G = np.where(M == 0)[0]

    # Partition mean and covariance
    m1_F, m1_G = m1[F], m1[G]
    m2_F, m2_G = m2[F], m2[G]

    y_F, y_G = y[F], y[G]

    S1_inv = inv(S1)
    S2_inv = inv(S2)

    # Partition inverse covariance matrices
    S1_FF = S1_inv[np.ix_(F, F)]
    S1_FG = S1_inv[np.ix_(F, G)]
    S1_GF = S1_inv[np.ix_(G, F)]
    S1_GG = S1_inv[np.ix_(G, G)]

    S2_FF = S2_inv[np.ix_(F, F)]
    S2_FG = S2_inv[np.ix_(F, G)]
    S2_GF = S2_inv[np.ix_(G, F)]
    S2_GG = S2_inv[np.ix_(G, G)]

    # Compute D and d
    D = S2_FF - S1_FF
    d = (S2_FF @ m2_F - S1_FF @ m1_F -
         (S2_FG @ (y_G - m2_G) - S1_FG @ (y_G - m1_G)))

    e = S2_FF @ m2_F - S1_FF @ m1_F
    f = (m2_F.T @ S2_FF @ m2_F) - (m1_F.T @ S1_FF @ m1_F)

    C = ((y_G - m2_G).T @ S2_GG @ (y_G - m2_G) -
         (y_G - m1_G).T @ S1_GG @ (y_G - m1_G))

    const = C - np.log(np.linalg.det(S1) / np.linalg.det(S2))

    def constraint_equation(lambda_):
        B = np.eye(len(F)) - lambda_ * D
        B_inv = inv(B)
        c = y_F - lambda_ * d

        term1 = c.T @ B_inv.T @ D @ B_inv @ c
        term2 = -2 * c.T @ B_inv.T @ e

        return term1 + term2 + f + const

    # Solve for lambda using numerical root-finding
    lambda_star = root(constraint_equation, x0=0.0).x[0]

    # Compute z_F
    B = np.eye(len(F)) - lambda_star * D
    z_F = inv(B) @ (y_F - lambda_star * d)

    # Combine free and fixed variables to get the full counterfactual
    z = np.copy(y)
    z[F] = z_F
    z[G] = y_G

    return z


def true_distance_midplane_cost_single_dist(X1, m1, m2):
    """
    Distance-to-midplane cost, as in your original implementation.
    """
    d = m1 - m2
    D2 = np.dot(d, d)
    if D2 == 0:
        return np.inf
    c = (m1 + m2) / 2
    d_unit = d / np.linalg.norm(d)
    dist1_sq = np.sum(((X1 - c) @ d_unit) ** 2)
    return dist1_sq


def cfq_score(X, labels_k, centers_k, covariances_k, log_likelihood):
    """
    Custom clustering criterion for Gaussian mixtures based on actionable counterfactuals.

    This is **exactly** your original logic:
    - compactness from squared distances to centers
    - for each cluster i, for each j != i:
        * build diagonal covariances from empirical variances of X_i and X_j
        * compute actionable counterfactuals from i toward j
        * accumulate squared distances ||z - y||^2
      then keep min cost over j
    - final score = total_sep / compactness
    """
    k = len(np.unique(labels_k))
    compactness = sum(
        np.sum((X[labels_k == j] - centers_k[j]) ** 2)
        for j in range(k)
    )

    min_sep_costs = []
    for i in range(k):
        costs = []
        for j in range(k):
            if i == j:
                continue

            X1 = X[labels_k == i]
            X2 = X[labels_k == j]
            m1 = centers_k[i]
            m2 = centers_k[j]
            S1 = covariances_k[i]
            S2 = covariances_k[j]
            M = np.ones(shape=X1.shape[1])

            cost = 0
            for y in X1:
                try:
                    diag_mat_1 = np.diag(np.var(X1, axis=0))
                    diag_mat_2 = np.diag(np.var(X2, axis=0))
                    z = compute_actionable_counterfactual(
                        m1, diag_mat_1, m2, diag_mat_2, y, M
                    )
                    # print('Normal case')
                except Exception:
                    # print('Except case')
                    identity_mat = np.eye(X1.shape[1])
                    z = compute_actionable_counterfactual(
                        m1, identity_mat, m2, identity_mat, y, M
                    )

                cost += np.linalg.norm(z - y) ** 2
            costs.append(cost)
        min_sep_costs.append(min(costs))

    total_sep = sum(min_sep_costs)
    mod_vrc_dist = total_sep / compactness if compactness > 0 else np.inf
    return mod_vrc_dist
