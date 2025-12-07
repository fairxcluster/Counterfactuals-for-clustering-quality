# Counterfactuals-for-clustering-quality
This repository provides functionality for **evaluating k-means and Gaussian Mixture Model (GMM) clustering solutions using the CFQ (Counterfactual-based Clustering Quality) score**.

The CFQ score is a clustering quality criterion grounded in **counterfactual reasoning**. Instead of relying solely on geometric separation or likelihood, it measures how *difficult it is to counterfactually move points across clusters*, relative to within-cluster compactness.

This repository is an **alternative and complementary framework** to counterfactual explanation generation: rather than explaining individual assignments, it **evaluates entire clustering solutions** through counterfactual distances.

## Overview

Given:
- A dataset
- A clustering solution (k-means or GMM)
- Cluster parameters (centroids, and optionally covariances)
  
Return:
**CFQ score**.

Higher CFQ values indicate clustering solutions where clusters better.

## Supported Models

The repository currently supports CFQ-based evaluation for:

### k-means clustering

- Uses cluster centers to define midplane decision boundaries
- Measures separation via **counterfactual distances** to the nearest cluster
- Compactness measured by within-cluster squared Euclidean distance
- Produces a score using counterfactual distances

### Gaussian Mixture Models (GMMs)

- Supports full-covariance Gaussian clusters
- Incorporates covariance structure and cluster geometry
- Separation is measured using **counterfactual distances** between Gaussian components
- Compactness measured via within-cluster squared distance to component means
- Produces a CFQ score grounded in counterfactual effort rather than likelihood alone

---

## CFQ Score Intuition

For both k-means and GMMs, the CFQ score follows the same principle:

- **Compactness:**  
  How tightly packed are points around their assigned cluster center?

- **Separation (counterfactual-based):**  
  How much change is required, on average, to move points into another cluster?

The score favors clustering solutions where **clusters are compact and well-separated**.

---

## How It Works

### k-means CFQ Evaluation

**File:** `kmeans_evaluator.py`

- For each cluster:
  - Compute within-cluster compactness as squared Euclidean distance to the center
- For each cluster pair:
  - Compute a midplane between centers
  - Measure how far points are projected from this midplane
- For each cluster, retain the *minimum* separation cost to any other cluster
- CFQ score = total separation / total compactness

This yields a geometry-based but counterfactual-inspired clustering quality measure.

---

### GMM CFQ Evaluation

**File:** `gmm_evaluator.py`

- Compactness:
  - Computed via squared distances of points to their assigned Gaussian mean
- Separation:
  - For each cluster pair `(i, j)`:
    - Generate actionable counterfactuals for points in cluster `i` toward cluster `j`
    - Counterfactuals respect covariance structure
    - Accumulate squared counterfactual distances
  - Retain minimal separation cost per cluster
- CFQ score = total counterfactual separation / total compactness

This results in a **counterfactual-driven evaluation of Gaussian clustering solutions**, incorporating both geometry and actionability.

---

## How to Run

This repository contains demonstration notebooks illustrating CFQ computation:

- **`CFQ for k-means (demo).ipynb`**
  - Demonstrates how to compute the CFQ score for a k-means clustering solution

- **`CFQ for GMMs (demo).ipynb`**
  - Demonstrates how to compute the CFQ score for a Gaussian mixture model

Typical workflow:
1. Fit a k-means or GMM model externally
2. Extract cluster labels and parameters
3. Pass them to the corresponding evaluator
4. Compute and analyze the CFQ score

---

### Reference
Georgios Vardakas, Antonia Karra, Evaggelia Pitoura, Aristidis Likas. "Evaluating Clustering Quality in Centroid-Based Clustering Using Counterfactual Distances." In International Conference on Discovery Science, pp. 222-236. Cham: Springer Nature Switzerland, 2025.


### Acknowledgments
The research project is implemented in the framework of H.F.R.I. call ``Basic research Financing (Horizontal support of all Sciences)'' under the National Recovery and Resilience Plan ``Greece 2.0'' funded by the European Union - NextGenerationEU (H.F.R.I. ProjectNumber: 15940).
