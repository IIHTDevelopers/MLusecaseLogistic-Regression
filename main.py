import pandas as pd
import numpy as np


# === Function 1: Load dataset ===
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# === Function 2: Preprocessing ===
def preprocess_data(df):
    # Impute missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Create target variable
    df['pass_fail'] = df['GPA'].apply(lambda gpa: 1 if gpa >= 3.0 else 0)

    # Drop unnecessary columns
    df = df.drop(columns=['GPA'], errors='ignore')
    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])

    return df


# === Function 3: Calculate UCL and LCL ===
def calculate_ucl_lcl(series):
    mean = series.mean()
    std = series.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    return ucl, lcl


# === Function 4: Sigmoid ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# === Function 5: Hypothesis ===
def hypothesis(X, weights):
    return sigmoid(np.dot(X, weights))


# === Function 6: Cost Function ===
def compute_cost(X, y, weights):
    m = len(y)
    predictions = hypothesis(X, weights)
    cost = - (1 / m) * np.sum(
        y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15)
    )
    return cost


# === Function 7: Predict pass/fail ===
def predict(X, weights):
    probs = hypothesis(X, weights)
    return ['Pass' if p >= 0.5 else 'Fail' for p in probs]


# === Main Program ===
def main():
    # 1. Load data
    df = load_data("Studentperformancedata.csv")  # Replace with your CSV file path

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. Calculate UCL/LCL
    ucl, lcl = calculate_ucl_lcl(df['StudyTimeWeekly'])
    print(f"UCL for StudyTimeWeekly: {ucl:.2f}")
    print(f"LCL for StudyTimeWeekly: {lcl:.2f}")

    # 4. Prepare features and labels
    feature_cols = ['StudyTimeWeekly', 'Absences', 'ParentalSupport']
    X = df[feature_cols].to_numpy().astype(float)
    y = df['pass_fail'].to_numpy().reshape(-1, 1).astype(float)

    # 5. Initialize weights
    weights = np.zeros((X.shape[1], 1))

    # 6. Compute cost
    cost = compute_cost(X, y, weights)
    print(f"\nInitial Cost (Log Loss): {cost:.4f}")
    print("Hypothesis: h(x) = 1 / (1 + e^-(β0 + β1*x1 + β2*x2 + ...))\n")

    # 7. Predict outcome for each student
    df['PredictedProbability'] = hypothesis(X, weights)
    df['Prediction'] = predict(X, weights)

    # 8. Show final output
    print("📊 Student Predictions:\n")
    if 'name' in df.columns:
        print(df[['name', 'PredictedProbability', 'Prediction']])
    else:
        print(df[['PredictedProbability', 'Prediction']])


# Run it
main()
