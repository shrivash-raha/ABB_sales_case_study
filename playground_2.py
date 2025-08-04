import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/train_v9rqX0R.csv')

df['Item_Weight'] = df['Item_Weight'].replace(np.nan, 'missing_data')

df['Item_weight_missing'] = (df['Item_Weight'] == 'missing_data').astype(int)

mean_weight = df.loc[df['Item_Weight'] != 'missing_data', 'Item_Weight'].astype(float).mean()
df['Item_Weight'] = df['Item_Weight'].replace('missing_data', mean_weight)

df['Outlet_Size'] = df['Outlet_Size'].replace(np.nan, 'missing_data')

df['UniqueID'] = df['Item_Identifier'].astype(str) + '_' + df['Outlet_Identifier'].astype(str)

df = df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

current_year = pd.Timestamp.now().year
df['Outlet_Age'] = current_year - df['Outlet_Establishment_Year']

df = df.drop('Outlet_Establishment_Year', axis=1)

cols = ['UniqueID', 'Item_Weight', 'Item_weight_missing', 'Outlet_Age']
rest = [col for col in df.columns if col not in cols + ['Item_Outlet_Sales']]
final_cols = cols + rest + ['Item_Outlet_Sales']
df = df[final_cols]

categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

cols_to_convert = df_dummies.columns.drop('UniqueID')
df_dummies[cols_to_convert] = df_dummies[cols_to_convert].apply(pd.to_numeric)

df_clean = df_dummies.drop('UniqueID', axis=1).copy()

X = df_clean.drop('Item_Outlet_Sales', axis=1)
y = df_clean['Item_Outlet_Sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print(importances)
# Item_weight_missing is least important. Need to find better way to handle missing values.
