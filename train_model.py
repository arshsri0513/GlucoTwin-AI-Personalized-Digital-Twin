import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def prepare_features(df):
    # Ensure data is sorted by patient and time
    df = df.sort_values(by=['patient_id', 'timestamp'])
    
    # Create lagged features to capture trends per patient
    for i in range(1, 5):
        df[f'glucose_lag_{i}'] = df.groupby('patient_id')['glucose_reading'].shift(i)
        df[f'activity_lag_{i}'] = df.groupby('patient_id')['activity_intensity'].shift(i)
        df[f'meal_lag_{i}'] = df.groupby('patient_id')['meal_carbs'].shift(i)
    
    # Glucose rate of change
    df['glucose_roc'] = df['glucose_reading'] - df['glucose_lag_1']
    
    # Rolling averages per patient
    df['glucose_rolling_mean_1h'] = df.groupby('patient_id')['glucose_reading'].transform(lambda x: x.rolling(window=4).mean())
    
    df.dropna(inplace=True)
    return df

def train():
    print("Loading data...")
    df = pd.read_csv('data/synthetic_glucose_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = prepare_features(df)
    
    # Features and target (patient_id is intentionally omitted from features)
    features = [
        'glucose_reading', 'hour', 'meal_carbs', 'activity_intensity',
        'glucose_lag_1', 'glucose_lag_2', 'glucose_lag_3', 'glucose_lag_4',
        'meal_lag_1', 'meal_lag_2', 'activity_lag_1', 'activity_lag_2',
        'glucose_roc', 'glucose_rolling_mean_1h'
    ]
    
    X = df[features]
    y = df['will_spike']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model on multi-patient data...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_glucose_model.pkl')
    print("Model saved to 'models/rf_glucose_model.pkl'")

if __name__ == "__main__":
    train()
