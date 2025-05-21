# Comparable-Property-Recommendation-System
This project builds an end-to-end system for recommending comparable properties in real estate appraisals. Trains a Random Forest to identify true comps and at inference, scores and returns the top matches with SHAP explanations.
A subject property
dataset https://github.com/H4mzaCode/CompRecommendation
A list of candidate properties

A set of true comps chosen by appraisers

The dataset is flattened into four structured DataFrames:
df_subjects
df_candidates
df_comps
df_appraisals

Data Preprocessing
All relevant numeric fields are cleaned and parsed (e.g., square footage, price). For each subject–candidate pair, we compute core features:
Difference in gross living area (GLA)
Age difference
Lot size difference
Price difference
Haversine distance
import numpy as np

##  Haversine Distance Formula

The Haversine formula is used to calculate the great-circle distance between two points on the Earth's surface given their latitudes and longitudes:

Where:

- `R` is the Earth's radius (≈ 6371 km)
- `φ₁`, `φ₂` are latitudes in radians
- `Δφ` is the difference in latitude: `φ₂ - φ₁`
- `Δλ` is the difference in longitude: `λ₂ - λ₁`

This formula accounts for the curvature of the Earth and was used to compute the geographic similarity between the subject and candidate properties.

## Haversine Distance Formula

$$
\text{distance} = 2R \cdot \arcsin\left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1) \cdot \cos(\phi_2) \cdot \sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)
$$

Where:

- \( R \): Earth’s radius (~6371 km)  
- \( \phi_1, \phi_2 \): Latitudes in radians  
- \( \Delta \phi = \phi_2 - \phi_1 \), \( \Delta \lambda = \lambda_2 - \lambda_1 \)




Structure type match (binary)
Bedroom count match (binary)

These features are used to label candidates as comp or not comp.

Model Training
A Random Forest Classifier is trained using an 80/20 stratified split with hyperparameters:

200 estimators
Max depth = 10
Minimum samples per leaf = 5
We handle edge cases where only one class is present.

Inference & Scoring
For any new subject, we:
Recompute features for all candidates Score them using the trained model

Rank and return the Top 3 recommended comps

Explainability
We use:
SHAP: Force plots visualize feature contributions for each top recommendation
LIME: Optional local interpretability for individual candidate explanations

Advanced Features
Clustering: K-Means applied on intrinsic features (GLA, lot size, year, etc.)
Similarity Search: Nearest Neighbors used to refine rankings

Visualizations: PCA plots, cluster bar charts, and map visualizations of comps vs. subjects
Sample Results
Top recommended comps returned with detailed scores
Cluster distributions and 2D scatter plots

Geospatial maps comparing subject and selected comps
Tech Stack
Python (pandas, sklearn, shap, lime, matplotlib, seaborn)


