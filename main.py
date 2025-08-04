import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv('data/train_v9rqX0R.csv')

# For Item_weight: Impute missing with 'missing_data'
df['Item_Weight'] = df['Item_Weight'].replace(np.nan, 'missing_data')

# Create dummy variable: Item_weight_missing
df['Item_weight_missing'] = (df['Item_Weight'] == 'missing_data').astype(int)

# Replace 'missing_data' with mean
mean_weight = df.loc[df['Item_Weight'] != 'missing_data', 'Item_Weight'].astype(float).mean()
df['Item_Weight'] = df['Item_Weight'].replace('missing_data', mean_weight)

# For outlet_size: Impute missing with 'missing_data'
df['Outlet_Size'] = df['Outlet_Size'].replace(np.nan, 'missing_data')


# Combine Item_Identifier and Outlet_Identifier to create UniqueID
df['UniqueID'] = df['Item_Identifier'].astype(str) + '_' + df['Outlet_Identifier'].astype(str)

# Drop the original columns
df = df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

# Calculate age from Outlet_Establishment_Year
current_year = pd.Timestamp.now().year
df['Outlet_Age'] = current_year - df['Outlet_Establishment_Year']

# Drop the original year column
df = df.drop('Outlet_Establishment_Year', axis=1)

# Restructure columns order
cols = ['UniqueID', 'Item_Weight', 'Item_weight_missing', 'Outlet_Age']
# Add remaining columns except Item_Outlet_Sales
rest = [col for col in df.columns if col not in cols + ['Item_Outlet_Sales']]
# Final order
final_cols = cols + rest + ['Item_Outlet_Sales']
df = df[final_cols]

# Create dummy variables for categorical columns (excluding UniqueID and target)
categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# Convert all columns except UniqueID to numeric
cols_to_convert = df_dummies.columns.drop('UniqueID')
df_dummies[cols_to_convert] = df_dummies[cols_to_convert].apply(pd.to_numeric)

df_clean = df_dummies.drop('UniqueID', axis=1).copy()

# Prepare X and y
X = df_clean.drop('Item_Outlet_Sales', axis=1)
y = df_clean['Item_Outlet_Sales']

# Fit model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print(importances)
# Item_weight_missing is least important. Need to find better way to handle missing values.



# I want to fit different models on the sales data I have. What should I try? It has a combination of categorical and numerical variables.
