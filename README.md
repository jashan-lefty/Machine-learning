# Spam Classification: Comparison and Interpretation of Results

## Overview
This project compares different classification algorithms on the Spambase dataset to determine which approach yields the highest accuracy. The algorithms tested include:
- K-Nearest Neighbours (KNN)
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)

This README outlines the chosen parameters, accuracy results, and observed outcomes of each classification method.

## Parameter Selection

### 1. K-Nearest Neighbours (KNN)
- Key Parameter: `n_neighbors`
- The model predicts a test point based on its `n_neighbors` closest points from the training dataset.
- Increasing `n_neighbors` may improve accuracy but too high a value can lead to false predictions.
- The highest accuracy was achieved with `n_neighbors = 5`.

### 2. Logistic Regression
- Logistic Regression models the probability of an input belonging to a certain class using a sigmoid function.
- **Key Parameter: `max_iter`**
  - Determines the maximum number of iterations before convergence.
  - If `max_iter` is too low, the model may not converge properly.
  - If too high, it may overfit.
- The highest accuracy was achieved with `max_iter = 100`.

### 3. Random Forest
- **Key Parameters:**
  - `n_estimators`: Number of decision trees in the forest.
  - `max_depth`: Depth of each decision tree.
  - `max_features`: Number of features used at each split.
- A higher number of trees improves accuracy but increases computational complexity.
- The best accuracy was obtained with `n_estimators = 100`.

### 4. Support Vector Machines (SVM)
- **Key Parameters:**
  - **Kernel:** Determines how the data is transformed into higher dimensions.
  - **C (Regularization Parameter):**
    - Low `C`: Larger margin, more misclassifications (underfitting).
    - High `C`: Smaller margin, fewer misclassifications (overfitting).
- The optimal `C` value was determined through experimentation.

## Accuracy Comparison
| Algorithm            | Accuracy (%) |
|----------------------|-------------|
| K-Nearest Neighbours | 89.27%       |
| Logistic Regression  | 90.80%       |
| Random Forest       | 93.48%       |
| Support Vector Machines | 92.39% |

## Findings & Conclusion
1. **Random Forest achieved the highest accuracy (93.48%)** due to its ensemble learning approach, which reduces overfitting and enhances generalization.
2. **SVM performed slightly lower (92.39%)** due to its capability to find non-linear decision boundaries, making it robust in complex datasets.
3. **Logistic Regression (90.80%)** showed strong performance due to its linear nature and standardization benefits.
4. **K-Nearest Neighbours had the lowest accuracy (89.27%)**, highlighting the impact of the `n_neighbors` parameter on its performance.

### Best Model: **Random Forest**
The combination of multiple decision trees in Random Forest makes it the most reliable model for spam detection, offering the best balance between accuracy and generalization.

---
By **Jashanpreet Singh**
S223028729  
SIT 384 – Task 6.3HD

