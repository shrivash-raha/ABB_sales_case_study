import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set float format to display numbers in normal digits
pd.options.display.float_format = '{:.2f}'.format

# Example DataFrame
df = pd.read_csv('data/train_v9rqX0R.csv')

# Create a histogram for 'Item_Visibility'
df['Item_Visibility'].plot.hist(bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.show()

# Total Sales in individual stores where products have more than 20% visibility
high_visibility_stores = df[df['Item_Visibility'] > 0.2]['Outlet_Identifier'].unique()

high_visibility_sales = df[df['Item_Visibility'] > 0.2].groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()
high_visibility_products = df[df['Item_Visibility'] > 0.2].groupby('Outlet_Identifier')['Item_Identifier'].nunique()

print("Stores with products having more than 20% visibility:")
print(high_visibility_stores)

print("\nNumber of unique products in stores with more than 20% visibility:")
print(high_visibility_products)

print("\nSales in these stores:")
print(high_visibility_sales)

# Group-wise sales for individual stores
grouped_sales = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
# Plotting the grouped sales
grouped_sales.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Total Sales by Outlet Identifier')
plt.xlabel('Outlet Identifier')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Group-wise total products for individual stores
grouped_products = df.groupby('Outlet_Identifier')['Item_Identifier'].nunique().sort_values(ascending=False)
# Plotting the grouped products
grouped_products.plot(kind='bar', color='green', edgecolor='black')
plt.title('Total Unique Products by Outlet Identifier')
plt.xlabel('Outlet Identifier')
plt.ylabel('Number of Unique Products')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Item_Type wise total sales
item_type_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
# Plotting the item type sales
item_type_sales.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Total Sales by Item Type')
plt.xlabel('Item Type')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Analysis on missing values in 'Item_Weight'
missing_item_weight = df[df['Item_Weight'].isnull()]
missing_item_weight['Outlet_Identifier'].value_counts()

# Display group-wise sales in normal digits
df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum().sort_values(ascending=False)

# Group-wise sales for missing item weight
missing_item_weight.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Outlet_Sales'].sum().sort_values(ascending=False)

# Average Item_Visibility for missing item weights and overall merged in single dataframe
merged_Item_Visibility = pd.concat([
    missing_item_weight.groupby(['Item_Type'])['Item_Visibility'].mean().sort_values(ascending=False),
    df.groupby(['Item_Type'])['Item_Visibility'].mean().sort_values(ascending=False)
], axis=1, keys=['Missing_Item_Weight_MRP', 'Overall_Item_Visibility'])



# Item_Identifiers with missing Item_Weight
missing_item_weight = df[df['Item_Weight'].isnull()]
available_item_weight = df[df['Item_Weight'].notnull()]

# Group by Item_Identifier, average Item_Weight, and count occurrences
available_item_weight_identifiers = available_item_weight.groupby('Item_Identifier').agg(
    Average_Item_Weight=('Item_Weight', 'mean'),
    Standard_Deviation=('Item_Weight', 'std'),
    Count=('Item_Weight', 'count')
).reset_index()

available_item_weight_identifiers # Item Weights are consistent for each Item_Identifier, with std_dev = 0



# Create a histogram for 'Item_MRP'
df['Item_MRP'].plot.hist(bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.show()

# Item_MRP with frequency less than 40
item_mrp_freq = df['Item_MRP'].value_counts()

# Item_MRP with Sales Scatter Plot
plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'], alpha=0.5, color='red')
plt.title('Item MRP vs Item Outlet Sales')
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')
plt.grid(True)
plt.show()


def outlet_size_sales_plot(input_df):
    # Data with available Outlet_Size
    outlet_size_data = input_df[input_df['Outlet_Size'].notnull()]
    missing_outlet_size = input_df[input_df['Outlet_Size'].isnull()]

    outlet_size_data[['Outlet_Size', 'Outlet_Identifier']].drop_duplicates()
    missing_outlet_size[['Outlet_Size', 'Outlet_Identifier']].drop_duplicates()

    # Total sales by Outlet_Identifier

    # Group by Outlet_Identifier, keep Outlet_Size (including missing), and calculate sales
    outlet_size_sales = input_df.groupby('Outlet_Identifier').agg(
        Outlet_Size=('Outlet_Size', 'first'),
        Total_Sales=('Item_Outlet_Sales', 'sum'),
        Average_Sales=('Item_Outlet_Sales', 'mean')
    ).reset_index()

    # Assign colors to Outlet_Size categories manually
    color_map = {
        'Small': 'red',
        'Medium': 'yellow',
        'High': 'green',
        None: 'gray',
        np.nan: 'gray'
    }

    # Map colors to each row, handling missing values
    bar_colors = outlet_size_sales['Outlet_Size'].apply(lambda x: color_map.get(x, 'gray'))

    # Plot
    plt.figure(figsize=(10,6))
    bars = plt.bar(outlet_size_sales['Outlet_Identifier'], outlet_size_sales['Total_Sales'], color=bar_colors, edgecolor='black')
    plt.xlabel('Outlet Identifier')
    plt.ylabel('Total Sales')
    plt.title('Total Sales by Outlet Identifier (Color-coded by Outlet Size)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create legend
    legend_labels = ['Small', 'Medium', 'High', 'None']
    legend_colors = ['red', 'yellow', 'green', 'gray']
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in legend_colors]
    plt.legend(handles, legend_labels, title='Outlet Size')

    return outlet_size_sales, plt

# Outlet Sales / Item MRP
outlet_sales = df.copy()
outlet_sales['Item_Outlet_Sales'] = outlet_sales['Item_Outlet_Sales'] / outlet_sales['Item_MRP']

outlet_sales_df, plot = outlet_size_sales_plot(outlet_sales)

plot.show()


# Outlet_Location_Type analysis
df['Outlet_Location_Type'].value_counts()
pd.crosstab(df['Outlet_Size'], df['Outlet_Location_Type'])

missing_outlet_size = df[df['Outlet_Size'].isnull()]
outlet_size_data = df[df['Outlet_Size'].notnull()]


# Outlet_Type analysis
outlet_size_data['Outlet_Type'].value_counts()
pd.crosstab(outlet_size_data['Outlet_Size'], outlet_size_data['Outlet_Type'])

missing_outlet_size = outlet_size_data[outlet_size_data['Outlet_Size'].isnull()]
missing_outlet_size['Outlet_Type'].unique()

# Outlet_Location_Type: Tier 1 & Tier3; and Outlet_Type: Supermarket Type1
missing_outlet_size[['Outlet_Location_Type', 'Outlet_Type']].value_counts()




# Sales data histogram
df['Item_Outlet_Sales'].plot.hist(bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Item Outlet Sales')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Frequency')
plt.show()






















