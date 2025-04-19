# Ensemble-based-property-price-prediction-using-stacking-neural-nets-SVR-boosting-feature-selection
This project predicts real estate prices using an ensemble of models like SVR, Random Forest, Gradient Boosting, and MLP. A Stacking Regressor combines their outputs, while Genetic Algorithm-based feature selection boosts performance and model accuracy.
---

##  Objective

To create a robust, high-performing property valuation model by:
- Leveraging multiple machine learning algorithms
- Optimizing feature selection
- Combining model predictions using stacked generalization

---

##  Technologies & Techniques

| Task                         | Tool / Algorithm                |
|------------------------------|----------------------------------|
| Base Models                  | Gradient Boosting, Random Forest, SVR, MLP |
| Meta Model                   | Linear Regression (Stacking Regressor)     |
| Feature Selection            | Genetic Algorithm (GeneticSelectionCV)     |
| Evaluation Metrics           | RMSE, R² Score, MAE              |
| Data Handling & Preprocessing| pandas, scikit-learn, NumPy      |
| Visualization                | matplotlib, seaborn              |

---

## Workflow Summary

1. **Data Preprocessing**
   - Cleaning and encoding features
   - Handling missing values and scaling

2. **Feature Optimization**
   - Genetic feature selection using `GeneticSelectionCV`
   - Reduced dimensionality while preserving predictive power

3. **Model Training**
   - Individual base models: SVR, RF, GradientBoost, MLPRegressor
   - Meta-model: Linear regression over predictions of base models (Stacking)

4. **Model Evaluation**
   - Metrics: RMSE, MAE, R² Score
   - Cross-validation to ensure model generalization

5. **Prediction & Visualization**
   - Final predictions vs. actual values
   - Feature importance and residuals

---

##  Sample Results

_(Replace with actual values from notebook)_

| Model                | RMSE   | R² Score |
|----------------------|--------|----------|
| Random Forest        | 25,400 | 0.88     |
| SVR                  | 28,100 | 0.84     |
| MLP Neural Net       | 27,900 | 0.85     |
| Gradient Boosting    | 24,700 | 0.89     |
| **Stacking Regressor** | **22,300** | **0.91** |

---

##  Key Features

- Robust **ensemble modeling** for improved accuracy
- Intelligent **feature selection** using genetic algorithms
- Scalable and adaptable to other regression problems
- Easy to extend with additional base models or feature engineering
