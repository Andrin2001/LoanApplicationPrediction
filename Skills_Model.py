################################################################################
# Machine Learning in Finance
# Predicting Loan Approvals
# Group 6
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import optuna
import pickle

# Step Guide
# Step 1: Data Cleaning and Preparation
#   Step 1.1: Create Dummy Variables
#   Step 1.2: Handle Missing Values
# Step 2: Random Forest Training & Feature Selection
#   Step 2.1: Split and Balance the Dataset
#   Step 2.2: Train Random Forest Model
#   Step 2.3: Feature Importance Evaluation
# Step 3: Hyperparameter Tuning with Optuna
#   Step 3.1: Define Objective Function
#   Step 3.2: Create and Optimize Optuna Study
# Step 4: Final Model Training & Evaluation
#   Step 4.1: Prepare Final Dataset
#   Step 4.2: Train Final Model
#   Step 4.3: Test Final Model & Save it

# Load data (use own path)
data_raw_reduced = pd.read_excel("ML_loan_data_raw.xlsx")

# Define a single RANDOM_STATE variable for reproducibility
RANDOM_STATE = 42

################################################################################
# Step 1: Data Cleaning and Preparation
################################################################################

# Step 1.1: Create Dummy Variables
# Create dummy variables for categorical features to use in machine learning models
data_clean = (
    data_raw_reduced.assign(
        immigrant=(data_raw_reduced["derived_ethnicity"] == "Hispanic or Latino").astype(int),
        white=(data_raw_reduced["derived_race"] == "White").astype(int),
        male=(data_raw_reduced["derived_sex"] == "Male").astype(int),
        loan_type_FHA=(data_raw_reduced["loan_type"] == "2").astype(int),
        loan_type_VA=(data_raw_reduced["loan_type"] == "3").astype(int),
        loan_type_RHS_or_FSA=(data_raw_reduced["loan_type"] == "4").astype(int),
        loan_purpose_HomePurchase=(data_raw_reduced["loan_purpose"] == "1").astype(int),
        loan_purpose_HomeImprovement=(data_raw_reduced["loan_purpose"] == "2").astype(int),
        first_lien=(data_raw_reduced["lien_status"] == "1").astype(int),
        open_end_loc=(data_raw_reduced["open_end_line_of_credit"] == "1").astype(int),
        private_purpose=(data_raw_reduced["business_or_commercial_purpose"] == "2").astype(int),
        interest_only_pay=(data_raw_reduced["interest_only_payment"] == "1").astype(int),
        balloon_pay=(data_raw_reduced["balloon_payment"] == "1").astype(int),
        site_built=(data_raw_reduced["construction_method"] == "1").astype(int),
        occupancy_type_second=(data_raw_reduced["occupancy_type"] == "2").astype(int),
        occupancy_type_investment=(data_raw_reduced["occupancy_type"] == "3").astype(int),
        debt_to_income_low=(data_raw_reduced["debt_to_income_ratio"] == "<20%").astype(int),
        debt_to_income_high=(data_raw_reduced["debt_to_income_ratio"] == ">60%").astype(int),
        applicant_age_young=(data_raw_reduced["applicant_age"].isin(["<25", "25-34"])).astype(int),
        applicant_age_old=(data_raw_reduced["applicant_age"].isin(["65-74", ">74"])).astype(int),
        initially_payable_to_inst=(data_raw_reduced["initially_payable_to_institution"] == "1").astype(int)
    )
    # Drop original columns that have been transformed into dummy variables
    .drop(
        columns=[
            "derived_ethnicity", "derived_race", "derived_sex", "loan_type", "loan_purpose",
            "lien_status", "open_end_line_of_credit", "business_or_commercial_purpose",
            "interest_only_payment", "balloon_payment", "construction_method",
            "occupancy_type", "debt_to_income_ratio", "applicant_age",
            "initially_payable_to_institution", "ffiec_msa_md_median_family_income",
            "tract_minority_population_percent", "census_tract", "tract_to_msa_income_percentage",
            "tract_owner_occupied_units", "tract_median_age_of_housing_units", "tract_population", "tract_one_to_four_family_homes", "white"
        ]
    )
)

# Ensure categorical variables are converted to the appropriate data type
categorical_columns = [
    "loan_approved", "immigrant", "male", "loan_type_FHA", 
    "loan_type_VA", "loan_type_RHS_or_FSA", "loan_purpose_HomePurchase", 
    "loan_purpose_HomeImprovement", "first_lien", "open_end_loc", 
    "private_purpose", "interest_only_pay", "balloon_pay", "site_built", 
    "occupancy_type_second", "occupancy_type_investment", "debt_to_income_low", 
    "debt_to_income_high", "applicant_age_young", "applicant_age_old", 
    "initially_payable_to_inst"
]
# Convert these columns to 'category' dtype for better memory usage and model performance
data_clean[categorical_columns] = data_clean[categorical_columns].astype("category")

# Display the structure of the cleaned dataset to understand its composition
print(data_clean.info())

# Step 1.2: Handle Missing Values
# Identify columns with missing values
missing_values = data_clean.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)

# Create missingness dummies for the specified columns to keep track of missing information
data_clean["property_value_missing"] = data_clean["property_value"].isnull().astype(int)
data_clean["combined_loan_to_value_ratio_missing"] = data_clean["combined_loan_to_value_ratio"].isnull().astype(int)

# Imputation using IterativeImputer to estimate missing values
imputer = IterativeImputer(random_state=RANDOM_STATE)
# Define columns to be imputed (excluding missingness dummies)
columns_to_impute = data_clean.drop(
    ["property_value_missing", "combined_loan_to_value_ratio_missing"], axis=1
).columns

# Perform imputation and create a new DataFrame with imputed values
imputed_array = imputer.fit_transform(data_clean[columns_to_impute])
data_imputed = pd.DataFrame(imputed_array, columns=columns_to_impute, index=data_clean.index)

# Add the missingness indicator columns back to the imputed DataFrame
data_imputed["property_value_missing"] = data_clean["property_value_missing"].astype("bool")
data_imputed["combined_loan_to_value_ratio_missing"] = data_clean["combined_loan_to_value_ratio_missing"].astype("bool")

# Re-convert binary and categorical columns to their correct types after imputation
data_imputed[categorical_columns] = data_imputed[categorical_columns].astype("bool")

# Check for any remaining missing values after imputation
missing_values = data_imputed.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)
print(data_imputed.info())

################################################################################
# Step 2: Random Forest Training & Feature Selection
################################################################################

# Step 2.1: Split and Balance the Dataset
# Split and balance the dataset
# Separate approved and denied loans to create a balanced dataset for training
approved = data_imputed[data_imputed["loan_approved"]]
denied = data_imputed[~data_imputed["loan_approved"]]
# Define a fixed sample size for both classes to ensure balance
sample_size = 50000
# Combine the sampled datasets and shuffle them to create a balanced dataset
data_balanced = pd.concat([
    approved.sample(n=sample_size, random_state=RANDOM_STATE),
    denied.sample(n=sample_size, random_state=RANDOM_STATE)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Output the class distribution to verify balance
print(data_balanced['loan_approved'].value_counts())

# Step 2.2: Train Random Forest Model
# Train-test split
# Split the balanced dataset into training and testing sets
X = data_balanced.drop(columns=["loan_approved"])
y = data_balanced["loan_approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Train a default Random Forest model
# Fit a Random Forest model with default hyperparameters
rf_default = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, oob_score=True, n_jobs=-1)
rf_default.fit(X_train, y_train)

# Evaluate the Out-of-Bag (OOB) score of the trained model
print(f"Out-of-Bag Score: {rf_default.oob_score_:.4f}")

# Step 2.3: Feature Importance Evaluation

# Perform permutation importance to evaluate the importance of each feature
perm_importance = permutation_importance(
    rf_default, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)

# Create a DataFrame to store feature importance values
perm_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# Define a threshold for dropping low-importance features
importance_threshold = 0.0000001  # You can adjust this threshold

# Split features into retained and dropped based on the importance threshold
retained_features = perm_importance_df[perm_importance_df["Importance"] >= importance_threshold]
dropped_features = perm_importance_df[perm_importance_df["Importance"] < importance_threshold]

# Retain only important features in the dataset for further modeling
X_train_reduced = X_train[retained_features["Feature"]]
X_test_reduced = X_test[retained_features["Feature"]]

# Print summaries of retained and dropped features
print("Retained Features:")
print(retained_features)

print("\nDropped Features:")
print(dropped_features)

################################################################################
# Step 3: Hyperparameter Tuning with Optuna
################################################################################

# Step 3.1: Define Objective Function
# Define the objective function for Optuna
# This function will be used to optimize hyperparameters for the Random Forest model
def objective(trial):
    # Suggest values for hyperparameters using Optuna
    n_estimators = trial.suggest_int("n_estimators", 200, 600, step=50)
    max_features = trial.suggest_int("max_features", 1, len(retained_features))
    
    # Initialize the Random Forest model with suggested hyperparameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True
    )
    
    # Train the model on the reduced feature set
    rf.fit(X_train_reduced, y_train)
    
    # Evaluate the model using accuracy on the test set
    y_pred = rf.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Step 3.2: Create and Optimize Optuna Study
# Create an Optuna study to maximize the model's accuracy
study = optuna.create_study(direction="maximize")
# Optimize the study using the defined objective function
study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)

# Extract the best parameters and accuracy score from the study
best_params = study.best_params
best_score = study.best_value

print(f"Best Parameters: {best_params}")
print(f"Best Accuracy: {best_score:.4f}")

################################################################################
# Step 4: Final Model Training & Evaluation
################################################################################

# Step 4.1: Prepare Final Dataset
# Train the final Random Forest model with the best hyperparameters
# Select the final feature set including the target variable
data_final = data_imputed[retained_features["Feature"].tolist() + ["loan_approved"]] # Keep target variable

# Split the final dataset into features and target
X_final = data_final.drop(columns=["loan_approved"])
y_final = data_final["loan_approved"]


# Split the final dataset into training and testing sets
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.2, random_state=RANDOM_STATE)

# Step 4.2: Train Final Model
# Initialize the final Random Forest model with optimized hyperparameters
final_rf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_features=best_params["max_features"],
    random_state=RANDOM_STATE,
    n_jobs=-1,
    oob_score=True,
    class_weight="balanced" # Optional: Handle class imbalance if needed
)

# Train the final Random Forest model
final_rf.fit(X_train_final, y_train_final)

# Step 4.3: Evaluate and Save Final Model
# Evaluate the model using the test set
y_pred_final = final_rf.predict(X_test_final)
final_accuracy = accuracy_score(y_test_final, y_pred_final)
print(f"Final Model Test Accuracy: {final_accuracy:.4f}")

# Evaluate the Out-of-Bag (OOB) score (only meaningful if bootstrap is True)
print(f"Final Model OOB Score: {final_rf.oob_score_:.4f}")

# Save the trained model to a file using pickle
with open("loan_approval_model.pkl", "wb") as file:
    pickle.dump(final_rf, file)
print("Model saved successfully as 'loan_approval_model.pkl'")