import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def show_model_data():
    """Display the data structure and how it's prepared for the LSTM model"""
    print("🔍 Coffee Roasting Model Data Analysis")
    print("=" * 60)
    
    # Load the data
    print("📊 Loading data files...")
    data_X = pd.read_csv('data_X.csv')
    data_Y = pd.read_csv('data_Y.csv')
    
    print(f"✅ Data loaded:")
    print(f"   - Sensor Data (data_X.csv): {data_X.shape}")
    print(f"   - Quality Data (data_Y.csv): {data_Y.shape}")
    
    # Show original data structure
    print(f"\n📋 Original Data Structure:")
    print("=" * 60)
    
    print(f"\n🔧 Sensor Data Columns (data_X.csv):")
    print(f"   - date_time: Timestamp column")
    print(f"   - T_data_1_1, T_data_1_2, T_data_1_3: Drying Zone temperatures")
    print(f"   - T_data_2_1, T_data_2_2, T_data_2_3: Pre-Roasting Zone temperatures")
    print(f"   - T_data_3_1, T_data_3_2, T_data_3_3: Main Roasting Zone temperatures")
    print(f"   - T_data_4_1, T_data_4_2, T_data_4_3: Post-Roasting Zone temperatures")
    print(f"   - T_data_5_1, T_data_5_2, T_data_5_3: Cooling Zone temperatures")
    print(f"   - H_data: Relative humidity")
    print(f"   - AH_data: Absolute humidity")
    
    print(f"\n🎯 Quality Data Columns (data_Y.csv):")
    print(f"   - date_time: Timestamp column")
    print(f"   - quality: Target quality score (221-505 range)")
    
    # Show sample of raw data
    print(f"\n📊 Sample of Raw Sensor Data (first 5 rows):")
    print("=" * 60)
    print(data_X.head())
    
    print(f"\n📊 Sample of Raw Quality Data (first 5 rows):")
    print("=" * 60)
    print(data_Y.head())
    
    # Data preprocessing steps
    print(f"\n🔄 Data Preprocessing Steps:")
    print("=" * 60)
    
    # Convert date_time to datetime
    data_X['date_time'] = pd.to_datetime(data_X['date_time'])
    data_Y['date_time'] = pd.to_datetime(data_Y['date_time'])
    
    # Merge data
    merged_data = pd.merge(data_X, data_Y, on='date_time', how='inner')
    print(f"✅ After temporal alignment: {merged_data.shape}")
    
    # Get feature columns
    feature_columns = [col for col in data_X.columns if col != 'date_time']
    features = merged_data[feature_columns].values
    targets = merged_data['quality'].values
    
    print(f"\n📈 Feature Analysis:")
    print("=" * 60)
    print(f"   - Number of features: {len(feature_columns)}")
    print(f"   - Feature names: {feature_columns}")
    print(f"   - Feature shape: {features.shape}")
    print(f"   - Target shape: {targets.shape}")
    
    # Show feature statistics
    print(f"\n📊 Feature Statistics (before cleaning):")
    print("=" * 60)
    feature_df = pd.DataFrame(features, columns=feature_columns)
    print(feature_df.describe())
    
    # Show target statistics
    print(f"\n🎯 Target Statistics:")
    print("=" * 60)
    print(f"   - Quality range: {targets.min():.1f} - {targets.max():.1f}")
    print(f"   - Mean quality: {targets.mean():.1f}")
    print(f"   - Std quality: {targets.std():.1f}")
    
    # Data cleaning demonstration
    print(f"\n🧹 Data Cleaning Process:")
    print("=" * 60)
    
    # Clean data (same as in the model)
    df = pd.DataFrame(features, columns=feature_columns)
    
    # Fix humidity
    humidity_cols = ['H_data', 'AH_data']
    for col in humidity_cols:
        if col in df.columns:
            high_humidity_mask = df[col] > 100
            df.loc[high_humidity_mask, col] = 100
            print(f"   Fixed {high_humidity_mask.sum()} humidity readings > 100% in {col}")
    
    # Remove negative temperatures
    temp_cols = [col for col in df.columns if 'T_data' in col]
    for col in temp_cols:
        negative_temp_mask = df[col] < 0
        df.loc[negative_temp_mask, col] = np.nan
        print(f"   Removed {negative_temp_mask.sum()} negative temperatures in {col}")
    
    # Remove extreme temperatures
    for col in temp_cols:
        extreme_temp_mask = df[col] > 800
        df.loc[extreme_temp_mask, col] = np.nan
        print(f"   Removed {extreme_temp_mask.sum()} extreme temperatures > 800°C in {col}")
    
    # Forward fill and drop NaN
    df = df.fillna(method='ffill').fillna(method='bfill')
    features_cleaned = df.values
    
    print(f"\n✅ After cleaning: {features_cleaned.shape}")
    
    # Show cleaned feature statistics
    print(f"\n📊 Feature Statistics (after cleaning):")
    print("=" * 60)
    cleaned_df = pd.DataFrame(features_cleaned, columns=feature_columns)
    print(cleaned_df.describe())
    
    # Data scaling
    print(f"\n📏 Data Scaling Process:")
    print("=" * 60)
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_scaled = scaler_X.fit_transform(features_cleaned)
    y_scaled = scaler_y.fit_transform(targets.reshape(-1, 1)).flatten()
    
    print(f"   - Features scaled using RobustScaler")
    print(f"   - Targets scaled using RobustScaler")
    print(f"   - Scaled features shape: {X_scaled.shape}")
    print(f"   - Scaled targets shape: {y_scaled.shape}")
    
    # Show scaled data statistics
    print(f"\n📊 Scaled Feature Statistics:")
    print("=" * 60)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    print(scaled_df.describe())
    
    # Sequence creation
    print(f"\n🔄 LSTM Sequence Creation:")
    print("=" * 60)
    
    time_steps = 24
    X_seq, y_seq = [], []
    
    for i in range(len(X_scaled) - time_steps):
        X_seq.append(X_scaled[i:(i + time_steps)])
        y_seq.append(y_scaled[i + time_steps])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"   - Time steps: {time_steps} (24 hours)")
    print(f"   - Sequence shape: {X_seq.shape}")
    print(f"   - Target sequence shape: {y_seq.shape}")
    print(f"   - Each sequence contains: {time_steps} time steps × {len(feature_columns)} features")
    
    # Show sequence structure
    print(f"\n📋 LSTM Input Structure:")
    print("=" * 60)
    print(f"   - Input shape: (batch_size, {time_steps}, {len(feature_columns)})")
    print(f"   - Example: (32, 24, 17) for batch_size=32")
    print(f"   - Each sample: 24 time steps × 17 sensor features")
    
    # Show a sample sequence
    print(f"\n🔍 Sample Sequence (first sequence):")
    print("=" * 60)
    print(f"   - Sequence shape: {X_seq[0].shape}")
    print(f"   - Time steps: {X_seq[0].shape[0]}")
    print(f"   - Features per time step: {X_seq[0].shape[1]}")
    
    # Show first few time steps of first sequence
    print(f"\n📊 First 3 time steps of first sequence:")
    print("=" * 60)
    sample_sequence = pd.DataFrame(X_seq[0][:3], columns=feature_columns)
    print(sample_sequence)
    
    # Show target for this sequence
    print(f"\n🎯 Target for first sequence:")
    print("=" * 60)
    target_scaled = y_seq[0]
    target_original = scaler_y.inverse_transform([[target_scaled]]).flatten()[0]
    print(f"   - Scaled target: {target_scaled:.4f}")
    print(f"   - Original target: {target_original:.1f}")
    
    # Data split
    print(f"\n✂️  Data Split:")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    print(f"   - Training set: {X_train.shape}")
    print(f"   - Test set: {X_test.shape}")
    print(f"   - Training targets: {y_train.shape}")
    print(f"   - Test targets: {y_test.shape}")
    
    # Model input summary
    print(f"\n🏗️  Model Input Summary:")
    print("=" * 60)
    print(f"   - Input shape: {X_train.shape[1:]}")
    print(f"   - Features: {len(feature_columns)} sensor readings")
    print(f"   - Time steps: {time_steps} hours")
    print(f"   - Target: 1 quality score")
    print(f"   - Data type: Float32")
    
    # Show feature importance by zone
    print(f"\n📍 Feature Mapping by Process Zone:")
    print("=" * 60)
    
    zone_mapping = {
        'Drying Zone': ['T_data_1_1', 'T_data_1_2', 'T_data_1_3'],
        'Pre-Roasting': ['T_data_2_1', 'T_data_2_2', 'T_data_2_3'],
        'Main Roasting': ['T_data_3_1', 'T_data_3_2', 'T_data_3_3'],
        'Post-Roasting': ['T_data_4_1', 'T_data_4_2', 'T_data_4_3'],
        'Cooling Zone': ['T_data_5_1', 'T_data_5_2', 'T_data_5_3'],
        'Humidity': ['H_data', 'AH_data']
    }
    
    for zone, sensors in zone_mapping.items():
        zone_data = cleaned_df[sensors]
        print(f"   {zone}:")
        print(f"     - Sensors: {sensors}")
        print(f"     - Range: {zone_data.min().min():.1f}°C - {zone_data.max().max():.1f}°C")
        print(f"     - Mean: {zone_data.mean().mean():.1f}°C")
    
    print(f"\n✅ Data Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    show_model_data()
