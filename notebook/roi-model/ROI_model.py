#!/usr/bin/env python
# coding: utf-8

# # Import Datasets
# Dataset 1: https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending
# <br>Dataset 2: https://www.kaggle.com/datasets/naniruddhan/online-advertising-digital-marketing-data

# In[83]:


import pandas as pd
import numpy as np
df1 = pd.read_csv('Marketing.csv')
df2 = pd.read_csv('archive/online_advertising_performance_data.csv')


# ### Data Cleaning for Dataset 1

# In[84]:


# Check for missing values and data types
print(df1.info())

# Drop duplicates
df1.drop_duplicates(inplace=True)

# Check for missing values
print(df1.isnull().sum())

# Drop missing values (if any)
df1 = df1.dropna()

# Calculate Click to Revenue Conversion Rate
df1['click_to_revenue_ratio'] = df1['revenue'] / df1['clicks']

df1_cleaned = df1.drop(columns=["id", "c_date", "campaign_name", "campaign_id"])
df1_cleaned = df1_cleaned.rename(columns={"mark_spent": "cost"})

# Verify cleaned data
print(df1_cleaned.head())


# ## Data Cleaning for Dataset 2

# In[85]:


# Check for missing values
print(df2.isnull().sum())

# Drop columns where all values are NaN
df2_cleaned = df2.dropna(axis=1, how='all')

# Verify the updated dataframe
print(df2_cleaned.columns)

# Define a base conversion rate (you can adjust this based on your understanding)
base_conversion_rate = 0.025  # 2.5% of clicks turn into leads

# Add random noise to simulate variance in campaign performance
conversion_adjustment = np.random.uniform(0.9, 1.1, len(df2))  # Random adjustment between 90% to 110%

# Create a synthetic 'leads' column based on clicks, budget, and random noise
df2_cleaned['leads'] = (df2_cleaned['clicks'] * base_conversion_rate * conversion_adjustment).astype(int)

# Remove the specified columns
df2_cleaned = df2_cleaned.drop(columns=["month", "day", "campaign_number", "user_engagement","banner"])

# Rename the 'placement' column to 'category'
df2_cleaned = df2_cleaned.rename(columns={"placement": "category"})
# Rename 'displays' to 'impressions'
df2_cleaned = df2_cleaned.rename(columns={"displays": "impressions"})
# Replace 'abc' with 'jkl' in the 'category' column
df2_cleaned['category'] = df2_cleaned['category'].replace('abc', 'jkl')
# Rename 'post_click_conversions' to 'orders'
df2_cleaned = df2_cleaned.rename(columns={"post_click_conversions": "orders"})

# Remove the 'revenue' column
df2_cleaned = df2_cleaned.drop(columns=["revenue"])

# Rename 'post_click_sales_amount' to 'revenue'
df2_cleaned = df2_cleaned.rename(columns={"post_click_sales_amount": "revenue"})
# Add a 'click_to_revenue_ratio' column, handling division by zero
df2_cleaned['click_to_revenue_ratio'] = df2_cleaned.apply(lambda row: row['revenue'] / row['clicks'] if row['clicks'] != 0 else 0, axis=1)

# Verify the new column
print(df2_cleaned[['revenue', 'clicks', 'click_to_revenue_ratio']].head())
# Replace category elements with new values
df2_cleaned['category'] = df2_cleaned['category'].replace({
    'mno': 'social',
    'def': 'search',
    'ghi': 'influencer',
    'jkl': 'media'
})

# Verify the changes
print(df2_cleaned['category'].value_counts())
print(df2_cleaned.head())


# ## Join Dataset 1 & Dataset 2

# In[86]:


common_columns = ["category", "impressions", "cost", "clicks", "leads", "orders", "revenue", "click_to_revenue_ratio"]

# Concatenate the two dataframes based on these common columns
df_combined = pd.concat([df1_cleaned[common_columns], df2_cleaned[common_columns]], axis=0, ignore_index=True)

# Verify the combined dataframe
print(df_combined.head())


# ## EDA

# ### Campaign Spending vs Revenue

# In[87]:


# Group by 'category' and calculate the correlation between 'cost' and 'revenue'
correlation_results = (
    df_combined
    .groupby('category')
    .apply(lambda x: x['cost'].corr(x['revenue']))
    .reset_index(name='correlation')
)

# Display the correlation results
print(correlation_results)


# In[88]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot Marketing Spend vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(x='cost', y='revenue', data=df_combined, hue='category', palette='coolwarm')
plt.title('Marketing Spend vs Revenue by Campaign Category')
plt.xlabel('Marketing Spend')
plt.ylabel('Revenue')
plt.show()


# ### Campaign Performance: Clicks, Leads, Orders

# In[89]:


metrics = ['clicks', 'leads', 'orders']

plt.figure(figsize=(10, 6))
sns.pairplot(df_combined[metrics])
plt.title('Clicks, Leads, and Orders Relationship')
plt.show()


# ### Revenue by Category

# In[90]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_combined, x='category', y='revenue', palette='Set3')
plt.title('Revenue Distribution by Campaign Category')
plt.xlabel('Campaign Category')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()


# ### Marketing Spend Distribution

# In[91]:


plt.figure(figsize=(10, 6))
sns.histplot(df_combined['cost'], bins=20, kde=True, color='green')
plt.title('Distribution of Marketing Spend')
plt.xlabel('Marketing Spend')
plt.ylabel('Frequency')
plt.show()


# ### Click to Revenue Conversion Rate

# In[92]:


# Plot Click to Revenue Ratio
plt.figure(figsize=(10, 6))
sns.histplot(df_combined['click_to_revenue_ratio'].dropna(), bins=20, kde=True, color='purple')
plt.title('Click to Revenue Conversion Ratio')
plt.xlabel('Click to Revenue Ratio')
plt.ylabel('Frequency')
plt.show()


# ### Revenue vs Leads

# In[93]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='leads', y='revenue', data=df_combined, hue='category', palette='coolwarm')
plt.title('Leads vs Revenue by Campaign Category')
plt.xlabel('Leads')
plt.ylabel('Revenue')
plt.show()


# # Random Forest Regressor

# In[94]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Assume df is your already available dataset
# Selecting features (category and cost) and targets (clicks, leads, orders)
X = df_combined[['category', 'cost']]  # Input features: category (categorical) and cost (numerical)
y = df_combined[['clicks', 'leads', 'orders']]  # Targets: clicks, leads, orders (all numerical)

# One-hot encoding for 'category'
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['category']])

# Concatenate the encoded category with cost
X_transformed = np.concatenate([X_encoded, X[['cost']].values], axis=1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Define and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R^2 score for each target
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Print the accuracy for each target variable
target_names = ['clicks', 'leads', 'orders']
for i, target in enumerate(target_names):
    print(f"{target} - MSE: {mse[i]}, R^2: {r2[i]}")

# ===========================
# Generating a Synthetic Dataset for Testing
# ===========================

# Categories for the synthetic data
categories = ['influencer', 'social', 'search', 'media']
n_samples = 100  # Number of synthetic data points

# Generating random synthetic data
synthetic_data = pd.DataFrame({
    'category': np.random.choice(categories, size=n_samples),
    'cost': np.random.uniform(100, 5000, size=n_samples)  # Random cost values
})

# One-hot encode the synthetic categories
synthetic_encoded = encoder.transform(synthetic_data[['category']])

# Concatenate the encoded categories with synthetic cost
X_synthetic = np.concatenate([synthetic_encoded, synthetic_data[['cost']].values], axis=1)

# Predict the target variables on synthetic data
y_synthetic_pred = model.predict(X_synthetic)

# Display the synthetic data and corresponding predictions
synthetic_data['clicks'] = y_synthetic_pred[:, 0]
synthetic_data['leads'] = y_synthetic_pred[:, 1]
synthetic_data['orders'] = y_synthetic_pred[:, 2]

# Show synthetic dataset with predictions
print(synthetic_data)

