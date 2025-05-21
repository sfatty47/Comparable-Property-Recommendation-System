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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(d_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

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


