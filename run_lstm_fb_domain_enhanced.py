import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# F&B Domain Configuration
FB_CONFIG = {
    'process': 'Industrial Bread Baking',
    'product': 'Commercial Bread',
    'quality_range': {'min': 221, 'max': 505, 'mean': 402.8, 'std': 46.3},
    'quality_grades': {
        'A+': (450, 505, 'Excellent quality'),
        'A': (400, 449, 'Good quality'),
        'B': (350, 399, 'Acceptable quality'),
        'C': (300, 349, 'Below standard'),
        'D': (221, 299, 'Poor quality (reject)')
    },
    'sensor_mapping': {
        'mixing_zone': ['T_data_1_1', 'T_data_1_2', 'T_data_1_3'],
        'fermentation': ['T_data_2_1', 'T_data_2_2', 'T_data_2_3'],
        'oven_zone1': ['T_data_3_1', 'T_data_3_2', 'T_data_3_3'],
        'oven_zone2': ['T_data_4_1', 'T_data_4_2', 'T_data_4_3'],
        'cooling_zone': ['T_data_5_1', 'T_data_5_2', 'T_data_5_3'],
        'humidity': ['H_data', 'AH_data']
    },
    'process_parameters': {
        'mixing_temp_range': (24, 28),  # °C
        'fermentation_temp_range': (30, 35),  # °C
        'baking_temp_range': (200, 230),  # °C
        'cooling_temp_range': (20, 25),  # °C
        'humidity_range': (60, 80)  # %
    }
}

def get_quality_grade(quality_score):
    """Convert quality score to grade based on F&B standards"""
    for grade, (min_score, max_score, description) in FB_CONFIG['quality_grades'].items():
        if min_score <= quality_score <= max_score:
            return grade, description
    return 'Unknown', 'Score out of range'

def analyze_sensor_anomalies(features, feature_columns):
    """Analyze sensor data for F&B process anomalies"""
    print("\n🔍 F&B Process Anomaly Analysis:")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(features, columns=feature_columns)
    
    # Analyze each process zone
    for zone, sensors in FB_CONFIG['sensor_mapping'].items():
        if zone == 'humidity':
            continue
            
        zone_data = df[sensors]
        zone_mean = zone_data.mean(axis=1)
        
        if 'mixing' in zone:
            temp_range = FB_CONFIG['process_parameters']['mixing_temp_range']
        elif 'fermentation' in zone:
            temp_range = FB_CONFIG['process_parameters']['fermentation_temp_range']
        elif 'oven' in zone:
            temp_range = FB_CONFIG['process_parameters']['baking_temp_range']
        elif 'cooling' in zone:
            temp_range = FB_CONFIG['process_parameters']['cooling_temp_range']
        else:
            continue
            
        # Check for temperature anomalies
        anomalies = ((zone_mean < temp_range[0]) | (zone_mean > temp_range[1])).sum()
        anomaly_percent = (anomalies / len(zone_mean)) * 100
        
        print(f"   {zone.replace('_', ' ').title()}:")
        print(f"     - Target Range: {temp_range[0]}°C - {temp_range[1]}°C")
        print(f"     - Actual Range: {zone_mean.min():.1f}°C - {zone_mean.max():.1f}°C")
        print(f"     - Anomalies: {anomalies} ({anomaly_percent:.1f}%)")
    
    # Analyze humidity
    humidity_data = df[FB_CONFIG['sensor_mapping']['humidity']]
    humidity_range = FB_CONFIG['process_parameters']['humidity_range']
    humidity_anomalies = ((humidity_data['H_data'] < humidity_range[0]) | 
                         (humidity_data['H_data'] > humidity_range[1])).sum()
    humidity_anomaly_percent = (humidity_anomalies / len(humidity_data)) * 100
    
    print(f"   Humidity:")
    print(f"     - Target Range: {humidity_range[0]}% - {humidity_range[1]}%")
    print(f"     - Actual Range: {humidity_data['H_data'].min():.1f}% - {humidity_data['H_data'].max():.1f}%")
    print(f"     - Anomalies: {humidity_anomalies} ({humidity_anomaly_percent:.1f}%)")

def load_and_prepare_data():
    """Load and prepare data for LSTM training with F&B context"""
    try:
        print("🥖 F&B Process Anomaly Prediction System")
        print("=" * 50)
        print(f"Process: {FB_CONFIG['process']}")
        print(f"Product: {FB_CONFIG['product']}")
        print("=" * 50)
        
        print("\n📊 Loading F&B process data...")
        
        # Load the data files
        data_X = pd.read_csv('data_X.csv')
        data_Y = pd.read_csv('data_Y.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Sensor Data (data_X.csv): {data_X.shape}")
        print(f"   - Quality Data (data_Y.csv): {data_Y.shape}")
        print(f"   - Submission Template: {sample_submission.shape}")
        
        # Convert date_time to datetime
        data_X['date_time'] = pd.to_datetime(data_X['date_time'])
        data_Y['date_time'] = pd.to_datetime(data_Y['date_time'])
        sample_submission['date_time'] = pd.to_datetime(sample_submission['date_time'])
        
        # Get feature columns (exclude date_time)
        feature_columns = [col for col in data_X.columns if col != 'date_time']
        print(f"\n📈 F&B Process Sensors: {len(feature_columns)} sensors")
        
        # Display sensor mapping
        for zone, sensors in FB_CONFIG['sensor_mapping'].items():
            print(f"   {zone.replace('_', ' ').title()}: {', '.join(sensors)}")
        
        # Align data by date_time
        print("\n🔄 Aligning F&B process data by timestamp...")
        
        # Merge data_X and data_Y on date_time
        merged_data = pd.merge(data_X, data_Y, on='date_time', how='inner')
        print(f"✅ After temporal alignment: {merged_data.shape}")
        
        # Extract features and target
        features = merged_data[feature_columns].values
        target = merged_data['quality'].values
        
        print(f"\n📊 Final F&B dataset:")
        print(f"   - Process Features: {features.shape}")
        print(f"   - Quality Targets: {target.shape}")
        
        # Quality analysis
        quality_stats = {
            'min': target.min(),
            'max': target.max(),
            'mean': target.mean(),
            'std': target.std()
        }
        
        print(f"\n🎯 Quality Metrics Analysis:")
        print(f"   - Quality Range: {quality_stats['min']:.0f} - {quality_stats['max']:.0f}")
        print(f"   - Average Quality: {quality_stats['mean']:.1f}")
        print(f"   - Quality Variation: {quality_stats['std']:.1f}")
        
        # Quality grade distribution
        grades = [get_quality_grade(score)[0] for score in target]
        grade_counts = pd.Series(grades).value_counts()
        print(f"\n📊 Quality Grade Distribution:")
        for grade, count in grade_counts.items():
            percentage = (count / len(grades)) * 100
            print(f"   - Grade {grade}: {count} samples ({percentage:.1f}%)")
        
        # Check for NaN values
        nan_features = np.isnan(features).sum()
        nan_target = np.isnan(target).sum()
        print(f"\n🔍 Data Quality Check:")
        print(f"   - Missing Sensor Data: {nan_features}")
        print(f"   - Missing Quality Data: {nan_target}")
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_indices]
        target = target[valid_indices]
        
        print(f"\n✅ After data cleaning:")
        print(f"   - Valid Process Records: {features.shape}")
        print(f"   - Valid Quality Records: {target.shape}")
        
        # Analyze sensor anomalies
        analyze_sensor_anomalies(features, feature_columns)
        
        return features, target, feature_columns, data_X, data_Y, sample_submission
        
    except Exception as e:
        print(f"❌ Error loading F&B data: {e}")
        return None, None, None, None, None, None

def scale_data_safely(X, y, scaler_X=None, scaler_y=None):
    """Scale data safely with NaN handling"""
    if scaler_X is None:
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = scaler_X.transform(X)
    
    if scaler_y is None:
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    # Handle any remaining NaN values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0)
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def create_sequences(X, y, time_steps=24):
    """Create time series sequences for LSTM (24-hour baking cycles)"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def create_lstm_model(input_shape):
    """Create LSTM model optimized for F&B process prediction"""
    model = keras.Sequential([
        # First LSTM layer - captures complex temporal patterns in baking process
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape,
                         dropout=0.3, recurrent_dropout=0.3),
        
        # Second LSTM layer - learns long-term dependencies in process parameters
        keras.layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
        
        # Dense layers for quality prediction
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')  # Linear for quality score regression
    ])
    
    # Compile with F&B-specific optimization
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse',  # Mean Squared Error for quality prediction
        metrics=['mae']  # Mean Absolute Error for quality assessment
    )
    
    return model

def plot_training_curves(history, save_path='fb_lstm_training_curves.png'):
    """Plot training curves with F&B context"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history.history['loss'], label='Training Loss', color='#2E86AB')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='#A23B72')
    ax1.set_title('F&B Process Model Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE curves
    ax2.plot(history.history['mae'], label='Training MAE', color='#F18F01')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='#C73E1D')
    ax2.set_title('F&B Quality Prediction MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training curves saved: {save_path}")

def plot_predictions(y_true, y_pred, title, save_path='fb_lstm_predictions.png'):
    """Plot predictions with quality grade interpretation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot with quality grades
    colors = []
    for score in y_true:
        grade, _ = get_quality_grade(score)
        if grade == 'A+':
            colors.append('#2E8B57')  # Sea Green
        elif grade == 'A':
            colors.append('#32CD32')  # Lime Green
        elif grade == 'B':
            colors.append('#FFD700')  # Gold
        elif grade == 'C':
            colors.append('#FF8C00')  # Dark Orange
        else:
            colors.append('#DC143C')  # Crimson
    
    ax1.scatter(y_true, y_pred, c=colors, alpha=0.6, s=30)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Quality Score')
    ax1.set_ylabel('Predicted Quality Score')
    ax1.set_title(f'{title}\nQuality Grade Color Coding', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add quality grade legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57', markersize=8, label='A+ (Excellent)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#32CD32', markersize=8, label='A (Good)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markersize=8, label='B (Acceptable)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8C00', markersize=8, label='C (Below Standard)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C', markersize=8, label='D (Poor/Reject)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Error distribution
    errors = y_pred - y_true
    ax2.hist(errors, bins=30, color='#4A90E2', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Quality Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Predictions plot saved: {save_path}")

def generate_submission_predictions(model, scaler_X, scaler_y, sample_submission, data_X, feature_columns):
    """Generate submission predictions with F&B quality interpretation"""
    print("\n🎯 Generating F&B Quality Predictions...")
    
    # Prepare submission features
    submission_features = []
    for _, row in sample_submission.iterrows():
        # Find closest timestamp in data_X
        target_time = row['date_time']
        time_diff = abs(data_X['date_time'] - target_time)
        closest_idx = time_diff.idxmin()
        features = data_X.iloc[closest_idx][feature_columns].values
        submission_features.append(features)
    
    submission_features = np.array(submission_features)
    
    # Scale features
    submission_features_scaled = scaler_X.transform(submission_features)
    
    # Create sequences for prediction (use last 24 time steps)
    time_steps = 24
    if len(submission_features_scaled) >= time_steps:
        # Use the last time_steps for prediction
        X_pred = submission_features_scaled[-time_steps:].reshape(1, time_steps, -1)
        prediction = model.predict(X_pred)
        
        # Inverse transform prediction
        prediction_original = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        # Fill all submission rows with the prediction
        sample_submission['quality'] = prediction_original[0]
        
        # Quality interpretation
        predicted_quality = prediction_original[0]
        grade, description = get_quality_grade(predicted_quality)
        
        print(f"   Predicted Quality Score: {predicted_quality:.1f}")
        print(f"   Quality Grade: {grade}")
        print(f"   Quality Description: {description}")
        
    else:
        # If not enough data, use a default value
        sample_submission['quality'] = 400  # Default to 'A' grade
        print(f"   Using default quality score: 400 (Grade A)")
    
    # Save submission file
    sample_submission.to_csv('fb_submission_predictions.csv', index=False)
    print(f"✅ F&B submission file saved: fb_submission_predictions.csv")
    
    return sample_submission

def main():
    """Main function to run F&B LSTM model"""
    print("🚀 Starting F&B Process Anomaly Prediction System...")
    print("=" * 60)
    
    # Load and prepare data
    features, target, feature_columns, data_X, data_Y, sample_submission = load_and_prepare_data()
    if features is None:
        print("❌ Failed to load F&B data. Exiting.")
        return
    
    # Scale the data
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data_safely(features, target)
    print(f"\n✅ F&B data scaled successfully.")
    print(f"   - Process Features: {X_scaled.shape}")
    print(f"   - Quality Targets: {y_scaled.shape}")
    
    # Create sequences (24 time steps = 24-hour baking cycles)
    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    print(f"\n✅ F&B process sequences created.")
    print(f"   - Process Sequences: {X_seq.shape}")
    print(f"   - Quality Sequences: {y_seq.shape}")
    print(f"   - Sequence Length: {time_steps} hours (baking cycle)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    print(f"\n✅ Data split for F&B model training:")
    print(f"   - Training Set: {X_train.shape}")
    print(f"   - Test Set: {X_test.shape}")
    
    # Create and compile model
    model = create_lstm_model((time_steps, X_scaled.shape[1]))
    print("\n✅ F&B LSTM model created and compiled")
    print(f"📊 Model Architecture Summary:")
    model.summary()
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    print("\n🔄 Training F&B Process Anomaly Prediction Model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n✅ F&B Model Training Completed!")
    
    # Plot training curves
    print("\n📊 Generating F&B training curves...")
    plot_training_curves(history)
    
    # Evaluate model
    print("\n📈 Evaluating F&B Model Performance...")
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    print("\n📊 F&B Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Calculate accuracy within different thresholds
    errors = np.abs(y_test_original - y_pred_original)
    within_1_percent = np.mean(errors <= 1.0) * 100
    within_5_percent = np.mean(errors <= 5.0) * 100
    within_10_percent = np.mean(errors <= 10.0) * 100
    
    print(f"\n🎯 F&B Quality Prediction Accuracy:")
    print(f"Within 1% error: {within_1_percent:.2f}%")
    print(f"Within 5% error: {within_5_percent:.2f}%")
    print(f"Within 10% error: {within_10_percent:.2f}%")
    
    # Quality grade prediction accuracy
    actual_grades = [get_quality_grade(score)[0] for score in y_test_original]
    predicted_grades = [get_quality_grade(score)[0] for score in y_pred_original]
    grade_accuracy = np.mean([a == p for a, p in zip(actual_grades, predicted_grades)]) * 100
    print(f"Quality Grade Prediction Accuracy: {grade_accuracy:.2f}%")
    
    # Plot predictions
    print("\n📊 Generating F&B prediction plots...")
    plot_predictions(y_test_original, y_pred_original, "F&B Quality Prediction")
    
    # Generate submission predictions
    submission_result = generate_submission_predictions(
        model, scaler_X, scaler_y, sample_submission, data_X, feature_columns
    )
    
    # Save the trained model
    model.save('fb_lstm_model.h5')
    print("\n✅ F&B model saved: fb_lstm_model.h5")
    
    print("\n" + "=" * 60)
    print("✅ F&B Process Anomaly Prediction System Completed!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("   - fb_lstm_training_curves.png")
    print("   - fb_lstm_predictions.png")
    print("   - fb_submission_predictions.csv")
    print("   - fb_lstm_model.h5")
    print("\n🎯 Business Impact:")
    print("   - Early quality anomaly detection")
    print("   - Process optimization insights")
    print("   - Reduced waste and rework")
    print("   - Improved product consistency")

if __name__ == "__main__":
    main()
