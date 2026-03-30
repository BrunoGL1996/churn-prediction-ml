# churn-prediction-ml

![Model Performance (ROC Curve)](02_curva_roc_final.png)

ROC Curve (Receiver Operating Characteristic)
"The ROC Curve is a performance measurement for classification problems at various threshold settings. The AUC (Area Under the Curve) of 0.81 indicates that our model has a high capability of distinguishing between customers who will churn and those who will stay. An AUC above 0.80 is considered an excellent result for business-driven predictive models, ensuring that our retention strategies are targeted at the right audience.

![Feature Importance (XGBoost)](03_feature_importance_final.png)

The Feature Importance plot identifies which variables most influenced the model's predictions. In this case, 'Contract' is the top predictor, suggesting that month-to-month contracts are the primary driver for churn. Other significant factors include 'Internet Service' and 'Online Security'. These insights allow the business to focus on high-impact interventions, such as incentivizing long-term contracts or improving technical support perceived value.


# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning and understanding the main factors behind it.

## Objective

- Predict which customers are more likely to churn  
- Identify the key variables influencing churn  
- Generate insights that can support business decisions  

## Model

- XGBoost classifier  
- AUC: ~0.81  

## Key Findings

- Customers on monthly contracts are more likely to churn  
- Lack of tech support is linked to higher churn  
- Manual payment methods show higher churn rates  

## Tech Stack

Python | Pandas | Scikit-learn | XGBoost | Matplotlib

## Notes

The goal of this project is not only to build a model, but also to understand the problem from a business perspective and suggest possible actions to reduce churn.

Machine learning model to predict customer churn and generate actionable business insights.
