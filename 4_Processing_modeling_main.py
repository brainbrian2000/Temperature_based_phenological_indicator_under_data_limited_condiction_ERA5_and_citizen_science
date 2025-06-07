import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold


# Load datasets into corresponding variables
flowering_df = pd.read_csv('spider_tree_flowering_converted.csv')
temperature_mean_df = pd.read_csv('temperature_previous_365days.csv')
temperature_GDD_df = pd.read_csv('temperature_previous_365days_min_max.csv')

# Fix column names to ensure 0–364 are strings
temperature_mean_df.columns = [str(i) for i in range(temperature_mean_df.shape[1])]
temperature_GDD_df.columns = [str(i) for i in range(temperature_GDD_df.shape[1])]

# Print the shape of each dataframe for verification
print("flowering_df shape:", flowering_df.shape)
print("temperature_mean_df shape:", temperature_mean_df.shape)
print("temperature_GDD_df shape:", temperature_GDD_df.shape)

# (Optional) Preview of the flowering records dataframe
flowering_df.head()

# Mean daily temperature over 365 days (retain columns '0' to '364')
temperature_mean_df_celsius = temperature_mean_df.copy()
numeric_cols_mean = [col for col in temperature_mean_df_celsius.columns
                     if col.isdigit() and 0 <= int(col) <= 364]

# Column check for daily mean temperature (0–364 days)
expected_mean_cols = set(str(i) for i in range(0, 365))
actual_mean_cols = set(numeric_cols_mean)
missing_mean_cols = expected_mean_cols - actual_mean_cols
if missing_mean_cols:
    print(f"Warning: Missing mean temperature columns: {sorted(missing_mean_cols)}")
else:
    print("Mean temperature columns from 0 to 364 are complete.")

temperature_mean_df_celsius[numeric_cols_mean] = temperature_mean_df_celsius[numeric_cols_mean] - 273.15

# Maximum/minimum daily temperatures over 365 days (retain columns '0' to '364').
temperature_GDD_df_celsius = temperature_GDD_df.copy()
numeric_cols_GDD = [col for col in temperature_GDD_df_celsius.columns
                    if col.isdigit() and 0 <= int(col) <= 364]

#Column check: GDD temperature
expected_GDD_cols = set(str(i) for i in range(0, 365))
actual_GDD_cols = set(numeric_cols_GDD)
missing_GDD_cols = expected_GDD_cols - actual_GDD_cols
if missing_GDD_cols:
    print(f"Warning: Missing GDD temperature columns: {sorted(missing_GDD_cols)}")
else:
    print("GDD temperature columns from 0 to 364 are complete.")

temperature_GDD_df_celsius[numeric_cols_GDD] = temperature_GDD_df_celsius[numeric_cols_GDD] - 273.15

# Show shapes after Celsius conversion for sanity check
print("temperature_mean_df_celsius shape:", temperature_mean_df_celsius.shape)
print("temperature_GDD_df_celsius shape:", temperature_GDD_df_celsius.shape)

# Convert the 'date' column to datetime format.
flowering_df['date'] = pd.to_datetime(flowering_df['date'])

# Add 'DOY' column (day of year)
flowering_df['DOY'] = flowering_df['date'].dt.dayofyear

# Check the results.
print(flowering_df[['date', 'DOY']].head())

# Confirm row alignment across the three dataframes
print("flowering_df rows:", flowering_df.shape[0])
print("temperature_mean_df_celsius rows:", temperature_mean_df_celsius.shape[0])
print("temperature_GDD_df_celsius rows:", temperature_GDD_df_celsius.shape[0])

# Horizontally concatenate flowering data with temperature data
# Use pd.concat(axis=1) to directly concatenate data horizontally (i.e., add columns).

# Merge flowering_df and temperature_mean_df_celsius.
merged_mean_df = pd.concat([flowering_df.reset_index(drop=True),
                            temperature_mean_df_celsius.reset_index(drop=True).iloc[:, 1:]], axis=1)

# Merge flowering_df and temperature_GDD_df_celsius
merged_GDD_df = pd.concat([flowering_df.reset_index(drop=True),
                           temperature_GDD_df_celsius.reset_index(drop=True).iloc[:, 1:]], axis=1)

# Results
print("merged_mean_df shape:", merged_mean_df.shape)
print("merged_GDD_df shape:", merged_GDD_df.shape)
merged_mean_df.head()

# Define target columns (day 1 to 180) for filtering
temp_cols = [str(i) for i in range(1, 181)]

# Filter merged_mean_df rows with positive temperature values
rows_to_keep_mean = merged_mean_df[temp_cols].sum(axis=1) > 0
merged_mean_df_cleaned = merged_mean_df[rows_to_keep_mean].reset_index(drop=True)

# Filter merged_GDD_df rows with positive temperature values
rows_to_keep_GDD = merged_GDD_df[temp_cols].sum(axis=1) > 0
merged_GDD_df_cleaned = merged_GDD_df[rows_to_keep_GDD].reset_index(drop=True)

# Results
print(f"merged_mean_df remain {merged_mean_df_cleaned.shape[0]} rows")
print(f"merged_GDD_df remain {merged_GDD_df_cleaned.shape[0]} rows")

# Select and retain only the daily temperature columns from day 0 to day 364
temp_cols = [col for col in merged_GDD_df_cleaned.columns
             if col.isdigit() and 0 <= int(col) <= 364]

# Verify completeness of the columns.
expected_cols = set(str(i) for i in range(0, 365))
actual_cols = set(temp_cols)
missing_cols = expected_cols - actual_cols
if missing_cols:
    print(f"Warning: Missing GDD columns (0–364): {sorted(missing_cols)}")
else:
    print("GDD columns from 0 to 364 are complete.")

# Compute GDD: subtract 15°C, set negatives to 0
merged_GDD_df_cleaned[temp_cols] = merged_GDD_df_cleaned[temp_cols].applymap(
    lambda x: max(x - 15, 0)
)

# Display partial results for inspection.
print("The data after GDD calculation (first few rows)::")
merged_GDD_df_cleaned[temp_cols].head()


# Define accumulation windows for GDD calculation
gdd_days_list = [90, 120, 180, 270, 365]

# Prepare to save the results.
best_thresholds = {}
best_r2s = {}

# Process different accumulation periods sequentially.
for days in gdd_days_list:
    # Select columns from day 0 to specified day threshold
    temp_cols = [col for col in merged_GDD_df_cleaned.columns
                 if col.isdigit() and 0 <= int(col) <= min(days, 364)]

    # Verify completeness of the columns.
    expected_cols = set(str(i) for i in range(0, min(days, 365)))
    actual_cols = set(temp_cols)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        print(f"Warning: Missing GDD columns for 0–{days} days: {sorted(missing_cols)}")
    else:
        print(f"GDD columns for 0–{days} are complete.")

    # Calculate accumulated GDD
    merged_GDD_df_cleaned[f'GDD_sum_0_{days}'] = merged_GDD_df_cleaned[temp_cols].sum(axis=1)

    # Filter records with DOY between 60 and 180
    filtered_df = merged_GDD_df_cleaned[
        (merged_GDD_df_cleaned['DOY'] >= 60) &
        (merged_GDD_df_cleaned['DOY'] <= 180)
    ]

    # Prepare X, Y
    X_DOY = filtered_df['DOY'].values
    Y_GDD = filtered_df[f'GDD_sum_0_{days}'].values

    # Set the search threshold range.
    threshold_range = np.arange(500, 3500, 5)

    best_r2 = -np.inf
    best_threshold = None
    r2_list = []

    for threshold in threshold_range:
        residual = Y_GDD - threshold
        residual = np.maximum(residual, 0)

        model = LinearRegression()
        model.fit(residual.reshape(-1, 1), X_DOY)
        preds = model.predict(residual.reshape(-1, 1))
        r2 = r2_score(X_DOY, preds)

        r2_list.append(r2)

        if r2 > best_r2:
            best_r2 = r2
            best_threshold = threshold

    # Store optimal threshold and R² for each period
    best_thresholds[days] = best_threshold
    best_r2s[days] = best_r2

    # Plot R² as a function of GDD threshold
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_range, r2_list, color='darkblue')
    plt.xlabel('GDD Threshold (°C-days)', fontsize=13)
    plt.ylabel('R² of DOY prediction', fontsize=13)
    plt.title(f'Maximizing R² to Estimate GDD Threshold ({days} days)', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Print the best results
print("===== Best GDD Thresholds and R² per period (DOY 60–180) =====")
for days in gdd_days_list:
    print(f"{days} days: Threshold = {best_thresholds[days]} °C-days, Best R² = {best_r2s[days]:.4f}")

    # Def of time scale
gdd_days_list = [90, 120, 180, 270, 365]

for days in gdd_days_list:
    # Limit the maximum column index to 364 to prevent misuse of column 365.
    max_col = min(days, 364)
    temp_cols = [str(i) for i in range(0, max_col + 1)]

    # Check the columns
    existing_cols = set(merged_GDD_df_cleaned.columns)
    missing_cols = set(temp_cols) - existing_cols
    if missing_cols:
        print(f"Warning: Missing columns for GDD_sum_0_{days}: {sorted(missing_cols)}")
    else:
        print(f"GDD_sum_0_{days} columns complete.")

    # Calculate GDD sum if not already present
    if f'GDD_sum_0_{days}' not in merged_GDD_df_cleaned.columns:
        merged_GDD_df_cleaned[f'GDD_sum_0_{days}'] = merged_GDD_df_cleaned[temp_cols].sum(axis=1)

    # Compute residual GDD as difference from optimal threshold
    merged_GDD_df_cleaned[f'GDD_residual_{days}d'] = (
        merged_GDD_df_cleaned[f'GDD_sum_0_{days}'] - best_thresholds[days]
    )

# Verify the newly added residual column.
print("Successfully added the Residual column! Current columns are:")
for days in gdd_days_list:
    print(f"GDD_residual_{days}d")

# Check partial data
merged_GDD_df_cleaned[[f'GDD_residual_{days}d' for days in gdd_days_list]].head()

# Define the number of days to process.
days_list = [90, 120, 180, 270, 365]

# Calculate temperature extremes (1%, 10% percentiles) for each period
for days in days_list:
    # Limit the maximum column index to 364 to prevent unintended use of column '365'.
    max_col = min(days, 364)
    temp_cols = [col for col in merged_mean_df_cleaned.columns
                 if col.isdigit() and 0 <= int(col) <= max_col]

    # Checking
    expected_cols = set(str(i) for i in range(0, max_col + 1))
    actual_cols = set(temp_cols)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        print(f"Warning: Missing mean temperature columns for 0–{days}: {sorted(missing_cols)}")
    else:
        print(f"Mean temperature columns for 0–{days} are complete.")

    # Extract temperature sequences
    temp_series = merged_mean_df_cleaned[temp_cols]

    # High temperature indicators
    merged_mean_df_cleaned[f'high1pct_temp_{days}d'] = temp_series.apply(
        lambda row: np.percentile(row, 99), axis=1
    )
    merged_mean_df_cleaned[f'high10pct_temp_{days}d'] = temp_series.apply(
        lambda row: np.percentile(row, 90), axis=1
    )

    # Low temperature indicators
    merged_mean_df_cleaned[f'low1pct_temp_{days}d'] = temp_series.apply(
        lambda row: np.percentile(row, 1), axis=1
    )
    merged_mean_df_cleaned[f'low10pct_temp_{days}d'] = temp_series.apply(
        lambda row: np.percentile(row, 10), axis=1
    )

# Check the added columns
print("Temperature threshold indicators added successfully! The newly added columns include:")
for days in days_list:
    print(f"high1pct_temp_{days}d, high10pct_temp_{days}d, low1pct_temp_{days}d, low10pct_temp_{days}d")
merged_mean_df_cleaned[[f'high1pct_temp_{days}d' for days in days_list]].head()

# Def of time scales
days_list = [90, 120, 180, 270, 365]

# Make sure mean temperature from day 0 to 'days' is computed for each sample
for days in days_list:
    max_col = min(days, 364)
    temp_cols = [col for col in merged_mean_df_cleaned.columns
                 if col.isdigit() and 0 <= int(col) <= max_col]

    # Verify column completeness and integrity
    expected_cols = set(str(i) for i in range(0, max_col + 1))
    actual_cols = set(temp_cols)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        print(f"Warning: Missing temperature columns for 0–{days}: {sorted(missing_cols)}")
    else:
        print(f"Columns for mean_temp_{days}d are complete.")

    # If mean temperature is not yet calculated, compute it now
    if f'mean_temp_{days}d' not in merged_mean_df_cleaned.columns:
        merged_mean_df_cleaned[f'mean_temp_{days}d'] = merged_mean_df_cleaned[temp_cols].mean(axis=1)

# Calculate the absolute difference between mean temperature and each threshold value.
for days in days_list:
    for temp_type in ['high1pct', 'high10pct', 'low1pct', 'low10pct']:
        mean_col = f'mean_temp_{days}d'
        threshold_col = f'{temp_type}_temp_{days}d'
        diff_col = f'diff_abs_mean_vs_{temp_type}_{days}d'

        merged_mean_df_cleaned[diff_col] = (
            (merged_mean_df_cleaned[mean_col] - merged_mean_df_cleaned[threshold_col]).abs()
        )

# Display the newly added columns.
print("Completed the calculation of differences between Mean Temperature and each threshold! The newly added columns are:")
for days in days_list:
    for temp_type in ['high1pct', 'high10pct', 'low1pct', 'low10pct']:
        print(f"diff_abs_mean_vs_{temp_type}_{days}d")


# Def of the time scales
periods = {
    '0_90': (0, 90),
    '90_180': (90, 180),
    '180_270': (180, 270),
    '270_365': (270, 365),
    '120_210': (120, 210),
    '210_300': (210, 300)
}

# Calculate standard deviation of temperature over defined intervals
for label, (start, end) in periods.items():
    # Limit the maximum value to 364 to avoid unintended use of column 365
    actual_end = min(end, 364)

    # Select columns with numeric names that fall within the specified range.
    cols = [col for col in merged_mean_df_cleaned.columns
            if col.isdigit() and start <= int(col) <= actual_end]

    # Column integrity check
    expected_cols = set(str(i) for i in range(start, actual_end + 1))
    actual_cols = set(cols)
    missing_cols = expected_cols - actual_cols

    if missing_cols:
        print(f"Warning: Missing columns for period {label}: {sorted(missing_cols)}")
    else:
        print(f"Columns for period {label} are complete.")

    # Calculate the standard deviation (if the columns exist).
    if cols:
        merged_mean_df_cleaned[f'std_temp_{label}'] = merged_mean_df_cleaned[cols].std(axis=1)
    else:
        print(f"Skipped period {label} due to empty column set.")

# Display completion message.
print("Standard deviations for each segment have been calculated and new columns added! The newly added columns are:")
for label in periods.keys():
    print(f'std_temp_{label}')


# === Assume you already have two cleaned datasets: merged_mean_df_cleaned and merged_GDD_df_cleaned ===

# Step 1: Define feature columns
mean_features = [
    col for col in merged_mean_df_cleaned.columns
    if (
        col.startswith('mean_temp_') or
        col.startswith('high1pct_temp_') or
        col.startswith('high10pct_temp_') or
        col.startswith('low1pct_temp_') or
        col.startswith('low10pct_temp_') or
        col.startswith('diff_abs_mean_vs_') or
        col.startswith('std_temp_')
    )
]

gdd_features = [
    col for col in merged_GDD_df_cleaned.columns
    if col.startswith('GDD_sum_') or col.startswith('GDD_residual_')
]

# Step 2: Merge the datasets and filter samples with DOY between 60 and 180.
merged_df = pd.merge(merged_mean_df_cleaned, merged_GDD_df_cleaned, on='id', suffixes=('_mean', '_gdd'))
filtered_df = merged_df[(merged_df['DOY_mean'] >= 60) & (merged_df['DOY_mean'] <= 180)]

# Step 3: Prepare the training data
X = filtered_df[mean_features + gdd_features]
y = filtered_df['DOY_mean']

# Step 4: Random Forest
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)

# Step 5: calculate feature importance
importances = rf.feature_importances_
features = X.columns.tolist()

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Step 6: Visualize feature importance for the top 30 features
plt.figure(figsize=(10, 12))
sns.set(style="whitegrid")
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(30),
    palette='viridis'
)

plt.title('Random Forest Feature Importance for Predicting DOY', fontsize=16, fontweight='bold')
plt.xlabel('Feature Importance (Gini Importance)', fontsize=13)
plt.ylabel('Features', fontsize=13)
plt.tight_layout()
plt.show()

# Step 1:First, filter samples with DOY between 60 and 180 using merged_df
# First, ensure merged_df is created as done in the Random Forest section
merged_df = pd.merge(merged_mean_df_cleaned, merged_GDD_df_cleaned, on='id', suffixes=('_mean', '_gdd'))
filtered_df = merged_df[(merged_df['DOY_mean'] >= 60) & (merged_df['DOY_mean'] <= 180)]

# Force conversion to a dense DataFrame to avoid merged_df being a csr_matrix.
merged_df = pd.DataFrame(merged_df.to_numpy(), columns=merged_df.columns)

# Step 2: Select the top 15 features ranked by Random Forest using the existing importance_df.
top15_features = importance_df.head(15)['Feature'].tolist()

# Step 3: Prepare X and y
X = filtered_df[top15_features]  # Now filtered_df contains all features
y = filtered_df['DOY_mean']  # Use 'DOY_mean' from merged_df

# Step 4: Features Standilization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: establish PLS model
pls = PLSRegression(n_components=2)
pls.fit(X_scaled, y)
y_pred = pls.predict(X_scaled).flatten()

# Step 6: Model performance
r2 = r2_score(y, y_pred)
# Remove the squared parameter or set it to True (default value)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"PLS (Top 15 Features after DOY filtering): R² = {r2:.4f}")
print(f"PLS (Top 15 Features after DOY filtering): RMSE = {rmse:.2f}")

# Step 7: Display prediction vs actual scatter plot
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred, s=80, color="#3498db", edgecolor="black", alpha=0.8)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')

plt.title('PLS Prediction vs Actual DOY (Top 15 Features, DOY 60–150)', fontsize=16, fontweight='bold')
plt.xlabel('Actual DOY', fontsize=13)
plt.ylabel('Predicted DOY', fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()


# Assume you already have y (actual DOY) and y_pred (initial PLS-predicted DOY)
# If y_pred is 2D, make sure it is flattened
# y_pred = y_pred.flatten()

# estabilish residual DataFrame
residuals_df = pd.DataFrame({
    'Actual DOY': y,
    'Predicted DOY': y_pred,
    'Residual': y - y_pred
})

# Plot R² as a function of GDD threshold
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")

sns.scatterplot(
    x='Predicted DOY',
    y='Residual',
    data=residuals_df,
    s=70,
    color='#e67e22',  
    edgecolor='black',
    alpha=0.8
)

plt.axhline(y=0, linestyle='--', color='black', linewidth=1.2)
plt.xlabel('Predicted DOY', fontsize=12)
plt.ylabel('Residual (Observed - Predicted)', fontsize=12)
plt.title('Supplementary Figure S1. Residual Plot of Initial PLS Model (Top 15 Features)', fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# remove the upper right frame
for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

# Save and Display
plt.savefig('supplementary_fig_s1_residual_initial_pls.png', dpi=300)
plt.show()


def calculate_vip(pls, X, y):
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    p, h = w.shape
    s = np.zeros((h,))
    for i in range(h):
        # Check if q is 1-dimensional or 2-dimensional and handle accordingly
        if q.ndim == 1:
            s[i] = np.sum((t[:, i] ** 2) * (q[i] ** 2))  # Use q[i] for 1D array
        else:
            s[i] = np.sum((t[:, i] ** 2) * (q[0, i] ** 2))  # Use q[0, i] for 2D array

    total_s = np.sum(s)
    vip_scores = np.zeros((p,))
    for j in range(p):
        weight = np.array([(w[j, i] ** 2) * s[i] for i in range(h)])
        vip_scores[j] = np.sqrt(p * np.sum(weight) / total_s)
    return vip_scores

# Calculate VIP score
vip_scores = calculate_vip(pls, X_scaled, y)


# Establish DataFrame
vip_df = pd.DataFrame({
    'Feature': top15_features,
    'VIP': vip_scores
}).sort_values(by='VIP', ascending=False)

# Display VIP bar chart
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")
sns.barplot(
    x='VIP',
    y='Feature',
    data=vip_df,
    palette='viridis',
    edgecolor='black'
)

plt.axvline(0.8, color='red', linestyle='--', linewidth=1.5, label='VIP = 0.8 threshold')
plt.title('VIP Scores for PLS Model (Top 15 Features)', fontsize=16, fontweight='bold')
plt.xlabel('Variable Importance in Projection (VIP)', fontsize=13)
plt.ylabel('Features', fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()
vip_df


# Step 1: Select features with VIP > 0.8
vip_selected_features = vip_df[vip_df['VIP'] > 0.8]['Feature'].tolist()

print(f"Number of features selected (VIP > 0.8): {len(vip_selected_features)}")
print(f"Selected features: {vip_selected_features}")

#Step 2: Prepare the data 
X = filtered_df[vip_selected_features]
y = filtered_df['DOY_mean']  # Change 'DOY' to 'DOY_mean'

# Step 3: standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4:  Train a new PLS model using selected features
pls_vip = PLSRegression(n_components=2)
pls_vip.fit(X_scaled, y)
y_pred_vip = pls_vip.predict(X_scaled).flatten()

# Step 5: Model Performance
r2_vip = r2_score(y, y_pred_vip)
# Instead of using squared=False, calculate RMSE manually if using an older scikit-learn version.
rmse_vip = np.sqrt(mean_squared_error(y, y_pred_vip))

print(f"Refined PLS (VIP > 0.8): R² = {r2_vip:.4f}")
print(f"Refined PLS (VIP > 0.8): RMSE = {rmse_vip:.2f}")

# Step 6: Display prediction vs actual scatter plot
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred_vip, s=80, color="#1abc9c", edgecolor="black", alpha=0.8)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')

plt.title('PLS Prediction vs Actual DOY (VIP > 0.8 Features)', fontsize=16, fontweight='bold')
plt.xlabel('Actual DOY', fontsize=13)
plt.ylabel('Predicted DOY', fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()


# Clean the data
residuals_df = pd.DataFrame({
    'Actual DOY': y, 
    'Predicted DOY': y_pred_vip, 
    'Residual': y - y_pred_vip
})

# Residual Plot
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")

sns.scatterplot(x='Predicted DOY', y='Residual', data=residuals_df,
                s=70, color='#2c7bb6', edgecolor='black', alpha=0.8)

plt.axhline(y=0, linestyle='--', color='black', linewidth=1.2)
plt.xlabel('Predicted DOY', fontsize=12)
plt.ylabel('Residual (Observed - Predicted)', fontsize=12)
plt.title('Residual Plot of Refined PLS Model', fontsize=15, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()


for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

plt.show()


# Step 1: Select the features with VIP > 0.8 from earlier.
vip_selected_features_08 = vip_df[vip_df['VIP'] > 0.8]['Feature'].tolist()

# Step 2: Use data with DOY in the range 60 to 180.
X = filtered_df[vip_selected_features_08]
# Change 'DOY' to 'DOY_mean' to match the column name in filtered_df
y = filtered_df['DOY_mean']

# Step 3: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Setup Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_list = []
rmse_list = []

# Step 5: Perform training and validation for each fold.
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pls_cv = PLSRegression(n_components=2)
    pls_cv.fit(X_train, y_train)
    y_pred = pls_cv.predict(X_test).flatten()

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_list.append(r2)
    rmse_list.append(rmse)

# Step 6: Results
mean_r2 = np.mean(r2_list)
mean_rmse = np.mean(rmse_list)

print(f"Cross-validated PLS (VIP > 0.8 Features)")
print(f"Average R² (5-fold CV): {mean_r2:.4f}")
print(f"Average RMSE (5-fold CV): {mean_rmse:.2f}")

# Step 1: Determine the coefficient
pls_coefficients = pls_vip.coef_.flatten()

# Step 2: Match corresponding feature names.
coef_table = pd.DataFrame({
    'Feature': vip_selected_features_08,
    'Coefficient': pls_coefficients
}).sort_values(by='Coefficient', ascending=False)

# Step 3: Dispaly
print(" PLS Regression Equation (standardized features):")
for feature, coef in zip(coef_table['Feature'], coef_table['Coefficient']):
    sign = "+" if coef >= 0 else "-"
    print(f" {sign} {abs(coef):.4f} * ({feature})")

print("\n Note: In PLS regression, the input features and target variable are usually mean-centered and scaled before fitting. As a result, the model does not include an explicit intercept term — the predicted values are centered around zero in the transformed space.")

coef_table


# Assume you already have vip_df, which is formatted as follows:
# vip_df = pd.DataFrame({'Feature': [...], 'VIP': [...], 'Coefficient': [...]})

# Filter variables with VIP > 0.8 along with their corresponding coefficients.
vip_selected = vip_df[vip_df['VIP'] > 0.8].copy()

# Merge coefficient information.
vip_selected = pd.merge(vip_selected, coef_table[['Feature', 'Coefficient']], on='Feature', how='left')

# Rearrange
vip_selected = vip_selected.sort_values(by='Coefficient', ascending=False)

# color setting
coef_colors = ['#4575b4' if x > 0 else '#d73027' for x in vip_selected['Coefficient']]

# Plot R² as a function of GDD threshold
plt.figure(figsize=(10, 6))
sns.set(style="white")

plt.bar(
    x=vip_selected['Feature'],
    height=vip_selected['Coefficient'],
    color=coef_colors,
    edgecolor='black'
)

# Plot Display
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylabel('Standardised Coefficient', fontsize=12)
plt.xlabel('Feature', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

plt.title('PLS Standardised Coefficients (VIP > 0.8)', fontsize=14, fontweight='bold')
plt.show()
