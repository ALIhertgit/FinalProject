Car Insurance Claim Prediction & Claim Amount Estimation
Project Overview
This project aims to predict whether a car insurance policyholder will file a claim (binary classification) and estimate the potential claim amount (regression). To achieve this, we leverage:
Machine Learning Pipelines for consistent data preprocessing.
Multiple Classification Models (Logistic Regression, KNN, Decision Tree, RandomForest, XGBoost, etc.) to compare accuracy.
Regression Models (RandomForestRegressor, XGBRegressor) to predict the value of a claim if filed.
Hyperparameter Tuning (RandomizedSearchCV, GridSearchCV) to optimize model performance.
The code handles data cleaning, feature engineering, model training, validation, and deployment strategies (by applying the pipeline to new data samples).
________________________________________
Dataset & Data Loading
car_insurance_claim.csv is loaded into a Pandas DataFrame raw_df. 
It contains demographic, vehicle, and historical claim data.
Duplicates are dropped.
Key steps:
Check duplicates: raw_df.duplicated().sum().
Drop duplicates: raw_df.drop_duplicates(inplace=True).
Initial shape: We log the shape of the dataset before and after cleaning.
________________________________________
Data Cleaning & Preprocessing
Renaming Columns
A dictionary (column_map) is used to rename cryptic column names into more descriptive ones (e.g., INCOME → income, CLAIM_FLAG → is_claim).
Dropping Unwanted Columns
We remove columns like ID and date_of_birth if they exist, as they are either unique identifiers or unnecessary for modeling.
Cleaning Currency Columns
Columns such as income, value_of_home, etc., may contain characters like “$” and “,”. The function clean_currency_values strips these and converts them to numeric (using .astype('Int64')).
Removing “z_” Prefixes
Some categorical columns (e.g., married, gender, highest_education) might have a “z_” prefix. The function strip_z_prefix removes that prefix via regex.
________________________________________
Exploratory Data Analysis (EDA)
Before creating the final pipeline, we explore:
Correlation with is_claim
We create eda_df from X_train and add back is_claim.
A correlation matrix (corr()) reveals which numeric features correlate with the target and each other.
Binning of new_claim_value
We create linear and log-scale bins (claim_value_cat, log_claim_value_cat) to understand the distribution of claim amounts.
Visualizations (not all shown in the code snippet, but recommended):
Histograms, boxplots, pairplots, correlation heatmaps, etc.
________________________________________
Splitting Data
The code creates:
X_all: All features except is_claim and new_claim_value.
y_all: The target is_claim.
A train/test split (train_test_split) is performed with stratify=X_all['claim_value_cat'] to ensure balanced distribution of claim bins.
________________________________________
Imputation (KNN & Simple)
KNN Imputer for numeric columns:
Looks at “neighboring” rows to fill missing numeric values.
SimpleImputer with most_frequent for categorical columns:
Replaces missing categories with the most common one.
The code organizes these steps into functions like:
knn_impute_numeric(...)
simple_impute_categorical(...)
________________________________________
Encoding Categorical Data
OrdinalEncoder for genuinely ordinal columns (e.g., highest_education).
Another OrdinalEncoder used for binary or yes/no style columns (like single_parent, gender).
OneHotEncoder for nominal columns (occupation, vehicle_type) to avoid imposing a false order.
The final “encoded” dataframe (X_train_encoded) is a combination of imputed numeric data, ordinal-encoded columns, binary-encoded columns, and one-hot-encoded columns.
________________________________________
Variance Inflation Factor (VIF)
We compute VIF on X_train_encoded to check for multicollinearity among features.
variance_inflation_factor from statsmodels is used.
If any features show extremely high VIF, they might be dropped.
________________________________________
Model Comparison (Classification)
We compare multiple classifier algorithms:
Logistic Regression, KNN, Decision Tree, RandomForest, XGBoost, AdaBoost, GradientBoost, Bagging, CatBoost.
Using 10-fold cross-validation (KFold), each model’s average accuracy is recorded and plotted via a boxplot. This helps us quickly see which models are performing well before we dive into advanced tuning.
________________________________________
Creating a Full Pipeline & Tuning (XGBoost Example)
ColumnTransformer + Pipeline
We build a scikit-learn Pipeline that:
Column Removal (e.g., dropping red_vehicle).
Numeric Pipeline (KNN → sqrt transform for skew → StandardScaler).
Ordinal + Binary + OneHot pipelines for different categorical columns.
The final XGBClassifier step.
Hyperparameter Tuning with RandomizedSearchCV
Parameters for XGB (n_estimators, max_depth, learning_rate) are sampled from distributions.
RandomizedSearchCV with f1_score (weighted) helps find the best settings while avoiding exhaustive search.
Evaluate on Test Set
Transform the X_test data with the best pipeline.
Predict y_test → measure the F1 score.
Show a Confusion Matrix for final classification performance.
________________________________________
Regression Pipeline
Filtering Rows
Only rows with new_claim_value > 0 are used for regression tasks (i.e., actual claim amounts).
Building a Similar Pipeline
We reuse the same full_preprocessor but replace the final step with a regressor (e.g., RandomForestRegressor or XGBRegressor).
We evaluate using RMSE and MAE (mean squared and absolute errors).
Hyperparameter Tuning (RandomizedSearchCV)
The code sets up a parameter grid (e.g., for XGBRegressor) and uses scoring='neg_root_mean_squared_error' to minimize RMSE.
The top candidates are displayed, and we can visualize with a bar chart.
Predicting on New Data
A single-row dictionary is created for the pipeline’s columns.
The pipeline’s .predict() method outputs the estimated claim amount.
________________________________________
Single New Data Sample Prediction
The code includes an example dictionary (e.g., sample_data) with all the columns the pipeline expects. Then:
sample_df = pd.DataFrame(sample_data)
best_model = rand_search.best_estimator_
sample_pred = best_model.predict(sample_df)
For classification: sample_pred[0] is 0 or 1.
For regression: It’s a numeric estimate.
