# IA Assignment 2 — Machine Learning Dataset Training

**Student:** B A UdayKiran  
**USN:** 1AM23CI020  
**Department:** CSE-AIML  
**Semester:** VII

## Dataset Mapping

| Task | Dataset | Algorithm |
|---|---|---|
| Classification | Social_Network_Ads.csv | Random Forest Classifier |
| Boosting | insurance_pre.csv | AdaBoost Regressor |
| Clustering | Mall_Customers.csv | K-Means |

## Results

- Classification accuracy: **90.00%**
- Boosting MAE: **3899.35**
- Boosting RMSE: **4925.61**
- Boosting R²: **0.8437**
- Clustering best k: **5**
- Clustering silhouette score: **0.5547**

## How to Run

Open `notebooks/IA_Assignment_2_ML_Datasets.ipynb` in Jupyter Notebook or Google Colab and run all cells.

The `datasets` folder contains the exact three datasets supplied for the assignment, and the `results` folder contains the generated result files and graphs.

## Important

The insurance target `charges` is continuous, so the boosting task is correctly treated as **regression** using AdaBoostRegressor.
