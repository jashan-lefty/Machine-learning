import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the Spambase dataset

data = pd.read_csv("spambase.csv")

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. K-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors can be tuned
knn.fit(X_train_scaled, y_train) #training the model
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbours Accuracy:", accuracy_knn)

# 2. Logistic Regression
log_reg = LogisticRegression(max_iter=100)  # Increase max_iter for convergence
log_reg.fit(X_train_scaled, y_train)   #training the model
y_pred_log_reg = log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", accuracy_log_reg)

# 3. Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # Can adjust n_estimators
random_forest.fit(X_train_scaled, y_train)  #training the model
y_pred_rf = random_forest.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# 4. Support Vector Machines (SVM)
svm = SVC(kernel='rbf', C=1)  # Can adjust C, kernel
svm.fit(X_train_scaled, y_train)   #training the model
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machines Accuracy:", accuracy_svm)
