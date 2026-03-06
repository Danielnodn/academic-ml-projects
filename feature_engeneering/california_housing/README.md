# California Housing Price Prediction

Author: Daniel Ramirez  

This project explores machine learning techniques to predict housing prices using the California Housing dataset.

The objective is to evaluate different regression models and determine which algorithm provides the best predictive performance for housing price estimation.

---
Machine learning workflow including:

- Data ingestion
- Feature engineering
- Model training
- Model comparison
- Performance evaluation
- Model export for potential deployment

The models evaluated in this analysis are:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Dataset

The dataset used in this project is the **California Housing Price dataset** : https://www.kaggle.com/datasets/camnugent/california-housing-prices

Main variables include:

- Median income
- Total rooms
- Total bedrooms
- Population
- Households
- Latitude and longitude
- Median house value (target variable)

The dataset is included in the repository inside the `sample_data` folder.

---

## Feature Engineering

Additional features were created to improve model performance:

- **rooms_per_household**  
  Average number of rooms per household.

- **bedrooms_ratio**  
  Proportion of bedrooms relative to total rooms.

- **population_per_household**  
  Population density within households.

Engineered features help capture relationships between housing characteristics and property value.

---

## Model Comparison

Three regression algorithms were evaluated:

| Model | RMSE | R² |
|------|------|------|
| Linear Regression | 68888 | 0.63 |
| Random Forest | 50980 | 0.79 |
| Gradient Boosting | 53672 | 0.77 |

### Results

Random Forest achieved the best performance with:

- **RMSE ≈ $50,980**
- **R² ≈ 0.79**

This indicates the model explains nearly **80% of the variance in housing prices**.

---

## Interpretation

The results suggest that housing prices are influenced by **non-linear relationships** between geographic and demographic features.

Tree-based ensemble models such as **Random Forest** and **Gradient Boosting** are better suited to capture these interactions compared to linear models.

---

## Visualization

The project includes visual comparisons of model performance as well as feature importance analysis to understand which variables contribute the most to price prediction.
