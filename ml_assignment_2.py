# IA Assignment 2 — ML Dataset Training
# Student: B A UdayKiran | USN: 1AM23CI020

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Classification
df = pd.read_csv("datasets/Social_Network_Ads.csv")
X = df[["Gender","Age","EstimatedSalary"]]
y = df["Purchased"]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["Gender"]),
    ("num", "passthrough", ["Age","EstimatedSalary"])
])
model = Pipeline([
    ("preprocessor", pre),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model.fit(Xtr,ytr)
pred = model.predict(Xte)
print("CLASSIFICATION ACCURACY:", accuracy_score(yte,pred))
print(classification_report(yte,pred))
print(confusion_matrix(yte,pred))

# 2. Boosting
df = pd.read_csv("datasets/insurance_pre.csv")
X = df.drop(columns=["charges"])
y = df["charges"]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["sex","smoker"]),
    ("num", "passthrough", ["age","bmi","children"])
])
model = Pipeline([
    ("preprocessor", pre),
    ("regressor", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=100, learning_rate=.05, random_state=42))
])
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.20,random_state=42)
model.fit(Xtr,ytr)
pred = model.predict(Xte)
print("BOOSTING MAE:", mean_absolute_error(yte,pred))
print("BOOSTING RMSE:", np.sqrt(mean_squared_error(yte,pred)))
print("BOOSTING R2:", r2_score(yte,pred))

# 3. Clustering
df = pd.read_csv("datasets/Mall_Customers.csv")
X = df[["Annual Income (k$)","Spending Score (1-100)"]]
X = StandardScaler().fit_transform(X)
scores={}
for k in range(2,11):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=km.fit_predict(X)
    scores[k]=silhouette_score(X,labels)
best_k=max(scores,key=scores.get)
km=KMeans(n_clusters=best_k,random_state=42,n_init=10)
labels=km.fit_predict(X)
print("CLUSTERING BEST K:",best_k)
print("CLUSTERING SILHOUETTE:",silhouette_score(X,labels))
