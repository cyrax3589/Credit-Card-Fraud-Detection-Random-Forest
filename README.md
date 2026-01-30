# Credit Card Fraud Detection using Random Forest

## Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques.  
The dataset used is highly imbalanced, making accuracy an unreliable metric. Hence, evaluation is performed using precision, recall, and F1-score.

The project compares a baseline Logistic Regression model with a Random Forest classifier to demonstrate the effectiveness of ensemble learning in fraud detection.

---

## Dataset

The dataset used for this project is the **Credit Card Fraud Detection Dataset** available on Kaggle.

🔗 Dataset Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

⚠️ **Note:**  
The dataset is **not uploaded to this repository** due to GitHub file size limitations.  
Please download the dataset manually from the link above and place the `creditcard.csv` file in the project root directory before running the notebook.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## Project Workflow

1. Load and explore the dataset
2. Analyze class imbalance
3. Split data using stratified sampling
4. Apply feature scaling
5. Train baseline Logistic Regression model
6. Train Random Forest classifier
7. Evaluate using:
   - Precision
   - Recall
   - F1-score
8. Plot feature importance
9. Save trained model

---

## Model Evaluation Metrics

The model performance is evaluated using:

- Precision
- Recall
- F1 Score
- Confusion Matrix

Accuracy is not used as a primary metric due to the highly imbalanced dataset.

---

## Feature Importance

Random Forest provides feature importance scores which help identify:
- Key transaction features contributing to fraud detection
- Patterns useful for risk assessment

<img width="851" height="547" alt="image" src="https://github.com/user-attachments/assets/4f4e9f9d-788e-4072-9c0c-551d3c54a90b" />


---

## Model Saving

The trained Random Forest model is saved using Joblib:

```bash
fraud_detection_model.pkl
```
This allows the model to be reused without retraining.

---

## What I Learned from This Project

Through this project, I learned why accuracy alone is misleading in fraud detection problems. Since fraudulent transactions represent only a very small fraction of total transactions, a model can achieve high accuracy by predicting every transaction as legitimate while still failing to detect fraud. This makes precision, recall, and F1-score far more meaningful for evaluating real-world performance.

I also gained a clear understanding of Random Forest as an ensemble learning algorithm. Instead of relying on a single model, Random Forest builds multiple decision trees and combines their predictions to produce more stable and accurate results. This approach reduces overfitting and improves generalization compared to individual models.

The concept of ensemble learning became clearer through implementation. By combining multiple weak or moderate learners, ensemble methods achieve better predictive power than standalone models. Random Forest is a practical example of this idea, using bagging and random feature selection to improve performance.

I learned the role of the `n_estimators` parameter, which controls the number of decision trees used in the Random Forest. Increasing this value generally improves performance and stability, but also increases training time and computational cost.

Finally, I understood the importance of SMOTE (Synthetic Minority Over-sampling Technique) in handling imbalanced datasets. SMOTE works by generating synthetic samples for the minority class instead of duplicating existing ones, helping the model learn meaningful patterns from fraudulent transactions and improving recall without causing severe overfitting.

