from pandasgui import show
import pandas as pd
import numpy as np

from data_cleaning import clean_data

data = pd.read_csv('data/train_v9rqX0R.csv')

# Clean the data
cleaned_data = clean_data(data, club_Item_Type=True, drop_outlet_size=False)
# show(cleaned_data)

# Remove identifiers
cleaned_data = cleaned_data.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

# Dummy variables for categorical columns
data_with_dummies = pd.get_dummies(cleaned_data, drop_first=True)
# show(data_with_dummies)


################# Model Training #################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

# Split features and target
X = data_with_dummies.drop('Item_Outlet_Sales', axis=1)
y = (data_with_dummies['Item_Outlet_Sales'])  # log1p handles zero values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Ridge Regression with tuning
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, ridge_params, cv=3, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
best_ridge = ridge_grid.best_estimator_

# Lasso Regression with tuning
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso = Lasso(max_iter=10000)
lasso_grid = GridSearchCV(lasso, lasso_params, cv=3, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
best_lasso = lasso_grid.best_estimator_

# Random Forest with tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_tuned = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf_tuned, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# XGBoost with tuning
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

xgb_tuned = XGBRegressor(random_state=42, n_jobs=-1)
xgb_grid = GridSearchCV(
    xgb_tuned,
    xgb_params,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

################# Model Evaluation #################
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val).clip(0)
    y_pred_actual = (y_pred)  # Convert back to original scale
    rmse = root_mean_squared_error((y_val), y_pred_actual)
    r2 = r2_score((y_val), y_pred_actual)
    mae = mean_absolute_error((y_val), y_pred_actual)
    mape = mean_absolute_percentage_error((y_val), y_pred_actual)
    return rmse, r2, mae, mape, y_pred, y_pred_actual

lr_metrics = evaluate_model(lr, X_val, y_val)
ridge_metrics = evaluate_model(best_ridge, X_val, y_val)
lasso_metrics = evaluate_model(best_lasso, X_val, y_val)
rf_tuned_metrics = evaluate_model(best_rf, X_val, y_val)
xgb_tuned_metrics = evaluate_model(best_xgb, X_val, y_val)

# DatfaFrame to hold metrics rounded to 2 decimal places
metrics_df = pd.DataFrame({
    'Model': [
        'Linear Regression', 'Ridge', 'Lasso',
        'Random Forest Tuned', 'XGBoost Tuned'
    ],
    'RMSE': [
        round(lr_metrics[0], 2), round(ridge_metrics[0], 2), round(lasso_metrics[0], 2),
        round(rf_tuned_metrics[0], 2), round(xgb_tuned_metrics[0], 2)
    ],
    'R2': [
        round(lr_metrics[1], 2), round(ridge_metrics[1], 2), round(lasso_metrics[1], 2),
        round(rf_tuned_metrics[1], 2), round(xgb_tuned_metrics[1], 2)
    ],
    'MAE': [
        round(lr_metrics[2], 2), round(ridge_metrics[2], 2), round(lasso_metrics[2], 2),
        round(rf_tuned_metrics[2], 2), round(xgb_tuned_metrics[2], 2)
    ],
    'MAPE': [
        round(lr_metrics[3], 2), round(ridge_metrics[3], 2), round(lasso_metrics[3], 2),
        round(rf_tuned_metrics[3], 2), round(xgb_tuned_metrics[3], 2)
    ]
})

# Predictions DataFrame in the form: Actual Sales, XGB, RF, ...
predictions_df = pd.DataFrame({
    'Actual Sales': (y_val),
    'XGBoost Tuned': (xgb_tuned_metrics[4]),
    'Random Forest Tuned': (rf_tuned_metrics[4]),
    'Ridge': (ridge_metrics[4]),
    'Lasso': (lasso_metrics[4])
})

print(metrics_df)
print(predictions_df)

################# Predicting Test Results #################
test_data = pd.read_csv('data/test_AbJTz2l.csv')
test_data_cleaned = clean_data(test_data, club_Item_Type=True, drop_outlet_size=False)
test_data_cleaned = test_data_cleaned.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
test_data_with_dummies = pd.get_dummies(test_data_cleaned, drop_first=True)

show(test_data_with_dummies)

# Predict using best_xgb
test_predictions = best_xgb.predict(test_data_with_dummies).clip(0)

# Prepare submission DataFrame
submission = pd.DataFrame({
    'Item_Identifier': test_data['Item_Identifier'],
    'Outlet_Identifier': test_data['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print(submission.head())


