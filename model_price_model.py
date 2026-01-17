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

# B2  Build the algorithm
####################################################


# Create the model
model = RandomForestRegressor(random_state=7)

print("Random forest regressor model created.")




# B3 Train the model
######################################################

print("Dataset shape:", df.shape)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=7
)

# Train the model
model.fit(X_train, y_train)

print("Model training complete.")

# B4 Evaluate the model's accuracy
#######################################################

# make predictions on the test data
predictions = model.predict(X_test)

# calculate evaluation metrics
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("Model Evaluation Results:")
print("R2 Score:", r2)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)



# B5  Cross-validation
#######################################################

# Apply cross validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

print("\nCross Validation Results:")
print("R2 scores for each fold:", cv_scores)
print("Average R2 score:", cv_scores.mean())




# B6 - Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV


param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

# Create grid search
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=7),
    param_grid,
    cv=5,
    scoring="r2"
)

# Train the grid search
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nHyperparameter results:")
print("Best params:", grid_search.best_params_)
