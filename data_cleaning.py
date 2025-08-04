from sklearn.preprocessing import StandardScaler

def clean_data(df, drop_outlet_size, club_Item_Type, current_year=2025):

    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(
        lambda x: x.fillna(x.mean())
    )
    # Fill remaining missing Item_Weight by Item_Type mean
    df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(
        lambda x: x.fillna(x.mean())
    )

    fat_map = {
        'Low Fat': 'Low Fat',
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'Regular': 'Regular',
        'reg': 'Regular'
    }
    df['Item_Fat_Content'] = df['Item_Fat_Content'].map(fat_map)

    if club_Item_Type:
        df['Item_Type'] = df['Item_Type'].apply(combine_item_types)

    if drop_outlet_size:
        df = df.drop('Outlet_Size', axis=1)
    else:
        df.loc[(df['Outlet_Size'].isna()) & (df['Outlet_Type'] == 'Supermarket Type1') & (df['Outlet_Location_Type'] == 'Tier 2'), 'Outlet_Size'] = 'Small'
        df['Outlet_Size'] = df['Outlet_Size'].fillna('Unknown')

    df['Outlet_Age'] = current_year - df['Outlet_Establishment_Year']
    df = df.drop('Outlet_Establishment_Year', axis=1)

    num_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def combine_item_types(row):
    if row in ['Fruits and Vegetables', 'Breakfast', 'Starchy Foods']:
        return 'Perishable'
    elif row in ['Canned', 'Frozen Foods', 'Baking Goods', 'Snack Foods', 'Dairy', 'Meat', 'Seafood', 'Breads']:
        return 'Processed Foods'
    elif row in ['Soft Drinks', 'Hard Drinks']:
        return 'Drinks'
    elif row in ['Health and Hygiene', 'Household']:
        return 'Non-Consumable'
    else:
        return 'Others'
