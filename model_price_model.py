# B1  Data preprocessing
#####################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error




# 1. Load the dataset
df = pd.read_csv("mobile_price_prediction_with_names.csv")

# 2. Check for missing values
print("Missing values:")
print(df.isnull().sum())

# 3. Convert  column (supports_5G) to numeric values (1 for True, 0 for False)
df["supports_5G"] = df["supports_5G"].astype(int)

# 4. Drop mobile_name because it is not useful for prediction
df = df.drop(columns=["mobile_name"])

# 5. Separate features and target variable
X = df.drop("price_USD", axis=1)
y = df["price_USD"]

# 6. Scale the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. Convert scaled features back to a dataframe
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 8. Combine features and target again
processed_df = pd.concat([X_scaled_df, y], axis=1)

# 9. Save the preprocessed dataset
processed_df.to_csv("preprocessed_mobile_data.csv", index=False)

print(" Data preprocessing complete. File saved as preprocessed_mobile_data.csv")

