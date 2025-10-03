# ML for Mining: Gold Recovery Prediction

"""
Gold Recovery Prediction Model for Mining Operations

This project develops machine learning models to predict gold recovery rates
in mining processing plants. The goal is to create regression models that
accurately predict recovery percentages in rougher and final stages using
sMAPE as the primary evaluation metric.

# Methodology:
# 1. Data loading and quality assessment
# 2. Recovery calculation verification
# 3. Data preprocessing and feature engineering
# 4. Exploratory data analysis
# 5. Model training and evaluation
# 6. Final model selection and validation
"""

# Visualization and analysis
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats as st
import matplotlib.pyplot as plt

# Data splitting and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, make_scorer
# Validation metrics
from sklearn.model_selection import cross_val_score

# Load datasets
df = pd.read_csv('/datasets/.gold_recovery_full.csv')
train = pd.read_csv('/datasets/.gold_recovery_train.csv')
test = pd.read_csv('/datasets/.gold_recovery_test.csv')

# Data visualization
# Full dataset
print('General information - Full dataset')
print('Dataset dimensions:', df.shape)
print('Column information:')
df.info()
print('\nData sample:')
df.head()

# Train dataset
print('\nGeneral information - Train dataset')
print('Dataset dimensions:', train.shape)
print('Column information:')
train.info()
print('\nData sample:')
train.head()

# Test dataset
print('\nGeneral information - Test dataset')
print('Dataset dimensions:', test.shape)
print('Column information:')
test.info()
print('\nData sample:')
test.head()


# Recovery calculation verification
# Define necessary columns
col_concentrate = 'rougher.output.concentrate_au'
col_feed = 'rougher.input.feed_au'
col_tail = 'rougher.output.tail_au'
col_target = 'rougher.output.recovery'

# Drop missing values
data_rec_calc = train[[col_concentrate, col_feed, col_tail, col_target]].dropna()

# Data with existing calculations
existing_recovery = data_rec_calc[col_target]

# Define calculation variables
c = data_rec_calc[col_concentrate]
f = data_rec_calc[col_feed]
t = data_rec_calc[col_tail]

# Formula
calculated_recovery = (c * (f - t)) / (f * (c - t)) * 100

# Mean Absolute Error between calculation and existing values
mae = round(mean_absolute_error(existing_recovery, calculated_recovery), 4)

# Display and compare results
print('Mean Absolute Error between calculated and existing values:', mae)
print('Comparison sample:')
rec_calc_comp = pd.DataFrame({
    'Existing Recovery': existing_recovery.head(),
    'Calculated Recovery': calculated_recovery.head(),
    'Absolute Difference': np.abs(round(existing_recovery.head() - calculated_recovery.head(), 4))
})
rec_calc_comp


# Data preprocessing and feature engineering
# Load columns from each dataset
train_cols = train.columns
test_cols = test.columns
print("Training columns:", len(train_cols))
print()
print("Test columns:", len(test_cols))

# Find missing columns in test set
missing_cols_set = set(train_cols).difference(set(test_cols))
missing_cols_list = list(missing_cols_set)

# Categorize missing columns
target_cols = []
concentrate_cols = []
tail_cols = []
calculation_cols = []
others = []
for col in missing_cols_list:
    if 'recovery' in col:
        target_cols.append(col)
    elif 'concentrate' in col:
        concentrate_cols.append(col)
    elif 'tail' in col:
        tail_cols.append(col)
    elif 'calculation' in col:
        calculation_cols.append(col)
    else:
        others.append(col)

# Display results
print(f"\nTotal missing columns in test set: {len(missing_cols_list)}")
print("\nMissing columns classification:")
print(f"Target columns: {len(target_cols)}, {target_cols}")
print(f"\nConcentrate columns: {len(concentrate_cols)}, {concentrate_cols}")
print(f"\nTail columns: {len(tail_cols)}, {tail_cols}")
print(f"\nCalculation columns: {len(calculation_cols)}, {calculation_cols}")
print(f"\nOthers: {len(others)}")

# Handle date column
df['date'] = pd.to_datetime(df['date'])
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
print(f"""Date column data type:
Full dataset: {df['date'].dtype}
Training dataset: {train['date'].dtype}
Test dataset: {test['date'].dtype}
""")

# Missing values
df_final = df.copy()
df_final = df.dropna(subset=['rougher.output.recovery', 'final.output.recovery'])

train_clean = train.copy()
numeric_train = train_clean.select_dtypes(include=[np.number]).columns
for col in numeric_train:
    if train_clean[col].isnull().sum() > 0:
        median_train = train_clean[col].median()
        train_clean[col].fillna(median_train, inplace=True)
print("\nNaNs in Train after imputation:", train_clean.isna().sum().sum())

test_clean = test.copy()
for col in numeric_train:
    if col in test_clean.columns:
        if test_clean[col].isnull().sum() > 0:
            median_from_train = train[col].median()
            test_clean[col].fillna(median_from_train, inplace=True)
print("NaNs in Test after imputation:", test_clean.isna().sum().sum())

# Duplicate handling
print('\nDuplicates in full dataset:', df.duplicated().sum())
print('Duplicates in train dataset:', train_clean.duplicated().sum())
print('Duplicates in test dataset:', test_clean.duplicated().sum())

# Feature alignment
common_features = test_clean.columns.tolist()
train_target_columns = ['rougher.output.recovery', 'final.output.recovery']
train_final = train_clean[common_features + train_target_columns].copy()
print("\nFinal training set:", train_final.shape)
print("Columns:", train_final.columns.tolist())


# Exploratory data analysis
# Gold boxplot
plt.figure(figsize=(10, 5))
au_data = [df_final['rougher.input.feed_au'], 
           df_final['rougher.output.concentrate_au'], 
           df_final['final.output.concentrate_au']]
sns.boxplot(data=au_data, palette='Paired')
plt.xticks([0, 1, 2], ['Feed', 'Rougher Concentrate', 'Final Concentrate'])
plt.title('Gold (Au) Concentration Distribution by Processing Stage')
plt.ylabel('Concentration (%)')
plt.show()

# Silver boxplot
plt.figure(figsize=(10, 5))
ag_data = [df_final['rougher.input.feed_ag'], 
           df_final['rougher.output.concentrate_ag'], 
           df_final['final.output.concentrate_ag']]
sns.boxplot(data=ag_data, palette='pastel')
plt.xticks([0, 1, 2], ['Feed', 'Rougher Concentrate', 'Final Concentrate'])
plt.title('Silver (Ag) Concentration Distribution by Processing Stage')
plt.ylabel('Concentration (%)')
plt.show()

# Lead boxplot
plt.figure(figsize=(10, 5))
pb_data = [df_final['rougher.input.feed_pb'].dropna(), 
           df_final['rougher.output.concentrate_pb'].dropna(), 
           df_final['final.output.concentrate_pb'].dropna()]
sns.boxplot(data=pb_data, palette="Set3")
plt.xticks([0, 1, 2], ['Feed', 'Rougher Concentrate', 'Final Concentrate'])
plt.title('Lead (Pb) Concentration Distribution by Processing Stage')
plt.ylabel('Concentration (%)')
plt.show()

# Training vs test histogram comparison
plt.figure(figsize=(10, 6))
plt.hist(train_final['rougher.input.feed_size'], bins=50, alpha=0.7, label='Train', color='pink')
plt.hist(test_clean['rougher.input.feed_size'], bins=50, alpha=0.7, label='Test', color='skyblue')
plt.title('Particle Size Distribution Comparison: Training vs Test Sets')
plt.xlabel('Particle Size (feed_size)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Total raw material concentration
df_final['total_feed'] = df_final[['rougher.input.feed_au', 'rougher.input.feed_ag', 'rougher.input.feed_pb', 'rougher.input.feed_sol']].sum(axis=1)

# Total rougher concentrate
df_final['total_rougher'] = df_final[['rougher.output.concentrate_au', 'rougher.output.concentrate_ag', 'rougher.output.concentrate_pb', 'rougher.output.concentrate_sol']].sum(axis=1)

# Total final concentrate
df_final['total_final'] = df_final[['final.output.concentrate_au', 'final.output.concentrate_ag', 'final.output.concentrate_pb', 'final.output.concentrate_sol']].sum(axis=1)

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=[df_final['total_feed'], 
                  df_final['total_rougher'], 
                  df_final['total_final']],
            palette="pastel")
plt.xticks([0, 1, 2], ['Raw Material', 'Rougher Concentrate', 'Final Concentrate'])
plt.title('Total Concentration Distribution by Processing Stage')
plt.ylabel('Total Concentration (%)')
plt.grid(True)
plt.show()

# Final concentration statistics
print('Concentration Statistics by Stage')
print('Raw Materials:')
print(df_final['total_feed'].describe())
print('\nRougher Concentrate:')
print(df_final['total_rougher'].describe())
print('\nFinal Concentrate:')
print(df_final['total_final'].describe())


# Model training and evaluation
# Individual sMAPE formula
def smape_individual(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    epsilon = 1e-10

    numerator = np.abs(y_real - y_pred)
    denominator = (np.abs(y_real) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(numerator / denominator) * 100

# Final sMAPE formula
def smape_final(y_real_rougher, y_pred_rougher, y_real_final, y_pred_final):
    smape_rougher_val = smape_individual(y_real_rougher, y_pred_rougher)
    smape_final_val = smape_individual(y_real_final, y_pred_final)
    return 0.25 * smape_rougher_val + 0.75 * smape_final_val

# Split data into training sets for rougher and final stages
X = train_final.drop(['rougher.output.recovery', 'final.output.recovery', 'date'], axis=1)
y_rougher = train_final['rougher.output.recovery']
y_final = train_final['final.output.recovery']

X_train, X_val, y_rougher_train, y_rougher_val = train_test_split(X, y_rougher, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_final_train, y_final_val = train_test_split(X, y_final, test_size=0.2, random_state=42, shuffle=True)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Verify data split
print('Total Samples:', len(train_final))
print(f'Training features: {len(X_train_scaled)} samples, 80% of data')
print(f'Validation features: {len(X_val_scaled)} samples, 20% of data')

# Defines score for cross validation
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
model_results = []

# Decision Tree Regressor function
def tree_model(model_name, X_train, y_train, X_val, y_val, scorer, concentrate_type):
    best_depth = 0
    best_val_mae = 0
    best_cross_val = float('inf')

    for depth in range(1, 21):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        model_predict = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, model_predict)

        cross_validation = cross_val_score(model, X_train, y_train, cv=3, scoring=scorer, n_jobs=-1)
        mean_cross_val = -cross_validation.mean()

        if mean_cross_val < best_cross_val:
            best_depth = depth
            best_val_mae = val_mae
            best_cross_val = mean_cross_val
            
    return {
        'Model': model_name,
        'Concentrate': concentrate_type,
        'Validation MAE': best_val_mae,
        'Depth': best_depth,
        'Cross Val Score': best_cross_val
    }

# Random Forest Regressor function
def forest_model(model_name, X_train, y_train, X_val, y_val, scorer, concentrate_type):
    best_depth = 0
    best_estimators = 0
    best_val_mae = 0
    best_cross_val = float('inf')

    for n_est in [100, 150]:
        for depth in [5, 10, 15]:
            model = RandomForestRegressor(max_depth=depth, n_estimators=n_est, random_state=42)
            model.fit(X_train, y_train)

            model_predict = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, model_predict)

            cross_validation = cross_val_score(model, X_train, y_train, cv=3, scoring=scorer, n_jobs=-1)
            mean_cross_val = -cross_validation.mean()

            if mean_cross_val < best_cross_val:
                best_depth = depth
                best_estimators = n_est
                best_val_mae = val_mae
                best_cross_val = mean_cross_val
            
    return {
        'Model': model_name,
        'Concentrate': concentrate_type,
        'Validation MAE': best_val_mae,
        'Depth': best_depth,
        'Estimators': best_estimators, 
        'Cross Val Score': best_cross_val
    }

# Linear Regression function
def linear_model(model_name, X_train, y_train, X_val, y_val, scorer, concentrate_type):
    best_val_mae = 0
    best_cross_val = float('inf')

    model = LinearRegression()
    model.fit(X_train, y_train)

    model_predict = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, model_predict)

    cross_validation = cross_val_score(model, X_train, y_train, cv=3, scoring=scorer, n_jobs=-1)
    mean_cross_val = -cross_validation.mean()

    if mean_cross_val < best_cross_val:
            best_val_mae = val_mae
            best_cross_val = mean_cross_val
            
    return {
        'Model': model_name,
        'Concentrate': concentrate_type,
        'Validation MAE': best_val_mae,
        'Cross Val Score': best_cross_val
    }

# DecisionTreeRegressor model for rougher stage
tree_rougher = tree_model('Decision Tree Regressor', 
                          X_train_scaled, y_rougher_train, 
                          X_val_scaled, y_rougher_val, 
                          mae_scorer, 
                          'Rougher'
                        )
model_results.append(tree_rougher)

# DecisionTreeRegressor model for final stage
tree_final = tree_model('Decision Tree Regressor', 
                          X_train_scaled, y_final_train, 
                          X_val_scaled, y_final_val, 
                          mae_scorer, 
                          'Final'
                        )
model_results.append(tree_final)

# LinearRegression model for rougher stage
linear_rougher = linear_model('Linear Regression',
                              X_train_scaled, y_rougher_train,
                              X_val_scaled, y_rougher_val,
                              mae_scorer,
                              'Rougher'
                              )  
model_results.append(linear_rougher)  

# LinearRegression model for final stage
linear_final = linear_model('Linear Regression',
                              X_train_scaled, y_final_train,
                              X_val_scaled, y_final_val,
                              mae_scorer,
                              'Final'
                              )  
model_results.append(linear_final)

# RandomForestRegressor model for rougher stage
forest_rougher = forest_model('Random Forest Regressor',
                              X_train_scaled, y_rougher_train,
                              X_val_scaled, y_rougher_val,
                              mae_scorer,
                              'Rougher'
                              )  
model_results.append(forest_rougher)  

# RandomForestRegressor model for final stage
forest_final = forest_model('Random Forest Regressor',
                              X_train_scaled, y_final_train,
                              X_val_scaled, y_final_val,
                              mae_scorer,
                              'Final'
                              )  
model_results.append(forest_final)  

# Results
df_results = pd.DataFrame(model_results)
df_results


# Final model selection and validation
# Test features and target
X_test = test_clean.drop('date', axis=1)

# Scaling
X_test_scaled = scaler.transform(X_test)

# Align full dataset samples with test set
y_rougher_real = df.loc[test_clean.index, 'rougher.output.recovery']
y_final_real = df.loc[test_clean.index, 'final.output.recovery']

# Filter NaN values
valid_index = y_rougher_real.notna() & y_final_real.notna()
y_rougher_valid = y_rougher_real[valid_index]
y_final_valid = y_final_real[valid_index]

# Rougher stage model
final_model_rougher = RandomForestRegressor(max_depth=15, n_estimators=150, random_state=42)
final_model_rougher.fit(X_train_scaled, y_rougher_train)

rougher_predictions = final_model_rougher.predict(X_test_scaled)

# Final stage model
final_model_final = RandomForestRegressor(max_depth=15, n_estimators=150, random_state=42)
final_model_final.fit(X_train_scaled, y_final_train)

final_predictions = final_model_final.predict(X_test_scaled)

# Clean predictions
rougher_pred_valid = rougher_predictions[valid_index]
final_pred_valid = final_predictions[valid_index]

# Calculate individual and final sMAPE
smape_calc_rougher = smape_individual(y_rougher_valid, rougher_pred_valid)
smape_calc_final = smape_individual(y_final_valid, final_pred_valid)
smape_total = smape_final(y_rougher_valid, rougher_pred_valid, y_final_valid, final_pred_valid)

# Results
final_model_results = pd.DataFrame({
    'Stage': ['Rougher', 'Final', 'Total'],
    'sMAPE': ([round(smape_calc_rougher, 2), round(smape_calc_final, 2), round(smape_total, 2)])
})

print('Final Results')
final_model_results