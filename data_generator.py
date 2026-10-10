import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import random

def generate_patient_data(patient_id, num_days=30):
    # Randomize patient info
    age = random.randint(35, 75)
    weight = random.randint(65, 110)
    gender = random.choice(['Male', 'Female'])
    hba1c = round(random.uniform(6.5, 9.5), 1)
    
    meds = random.choice(['Metformin 500mg', 'Metformin 1000mg', 'Glipizide 5mg', 'None'])
    
    patient_info = {
        'id': patient_id,
        'age': age,
        'gender': gender,
        'weight_kg': weight,
        'height_cm': random.randint(155, 185),
        'hba1c': hba1c,
        'diagnosis': 'Type 2 Diabetes',
        'medication': meds
    }

    # Generate time series data (every 15 minutes)
    start_time = datetime.now() - timedelta(days=num_days)
    times = [start_time + timedelta(minutes=15*i) for i in range(num_days * 24 * 4)]
    
    df = pd.DataFrame({'timestamp': times})
    df['patient_id'] = patient_id
    
    # Base glucose level with diurnal variation
    df['hour'] = df['timestamp'].dt.hour
    
    # Higher HbA1c roughly corresponds to higher average glucose
    base_gl = 100 + (hba1c - 6.0) * 15
    df['base_glucose'] = base_gl + 20 * np.sin(np.pi * (df['hour'] - 8) / 12)
    
    # Meals (Carbs intake)
    df['meal_carbs'] = 0.0
    for i in range(len(df)):
        h = df.loc[i, 'hour']
        m = df.loc[i, 'timestamp'].minute
        if h == 8 and m == 0:
            df.loc[i, 'meal_carbs'] = max(0, np.random.normal(45, 10))
        elif h == 13 and m == 0:
            df.loc[i, 'meal_carbs'] = max(0, np.random.normal(60, 15))
        elif h == 19 and m == 0:
            df.loc[i, 'meal_carbs'] = max(0, np.random.normal(50, 10))
    
    # Activity (Steps/Intensity)
    df['activity_intensity'] = 0.0 # 0 to 1
    active_hour = random.randint(16, 19) # Randomize primary active hour per patient
    for i in range(len(df)):
        h = df.loc[i, 'hour']
        if active_hour <= h <= active_hour + 1:
            df.loc[i, 'activity_intensity'] = np.random.beta(2, 5)
        else:
            df.loc[i, 'activity_intensity'] = max(0, np.random.normal(0.05, 0.05))
            
    # Calculate Glucose impact
    df['glucose_reading'] = df['base_glucose'] + np.random.normal(0, 5, len(df))
    
    # Add meal impact
    for i in range(len(df)):
        if df.loc[i, 'meal_carbs'] > 0:
            carbs = df.loc[i, 'meal_carbs']
            for j in range(1, 9):
                if i + j < len(df):
                    impact = carbs * 0.8 * (1 - abs(j - 4)/4)
                    df.loc[i+j, 'glucose_reading'] += max(0, impact)
                    
    # Reduce glucose based on activity
    for i in range(len(df)):
        if df.loc[i, 'activity_intensity'] > 0.3:
            intensity = df.loc[i, 'activity_intensity']
            for j in range(1, 5):
                if i + j < len(df):
                    df.loc[i+j, 'glucose_reading'] -= intensity * 15

    # Ensure realistic bounds
    df['glucose_reading'] = df['glucose_reading'].clip(60, 350)
    
    # Target: Glucose spike (>= 180) in the NEXT 2 hours
    df['future_max_glucose'] = df['glucose_reading'].rolling(window=8, min_periods=1).max().shift(-8)
    df['will_spike'] = (df['future_max_glucose'] >= 180).astype(int)
    
    return patient_info, df

if __name__ == "__main__":
    print("Generating synthetic data for multiple patients...")
    all_patients = []
    all_data = []
    
    # Generate 5 diverse patients
    patient_ids = ['P001', 'P002', 'P003', 'P004', 'P005']
    
    for pid in patient_ids:
        p_info, p_df = generate_patient_data(pid, num_days=30)
        all_patients.append(p_info)
        all_data.append(p_df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/synthetic_glucose_data.csv', index=False)
    
    with open('data/patient_profiles.json', 'w') as f:
        json.dump(all_patients, f)
        
    print("Data generated in 'data' directory.")
