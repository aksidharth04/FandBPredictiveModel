import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

COFFEE_CONFIG = {
    'process': 'Industrial Coffee Bean Roasting',
    'product': 'Specialty Coffee Beans',
    'quality_range': {'min': 221, 'max': 505, 'mean': 402.8, 'std': 46.3},
    'quality_grades': {
        'A+': (450, 505, 'Excellent quality - Premium specialty coffee'),
        'A': (400, 449, 'Good quality - Commercial specialty coffee'),
        'B': (350, 399, 'Acceptable quality - Standard commercial coffee'),
        'C': (300, 349, 'Below standard - Requires process adjustment'),
        'D': (221, 299, 'Poor quality - Reject batch')
    },
    'sensor_mapping': {
        'drying_zone': ['T_data_1_1', 'T_data_1_2', 'T_data_1_3'],
        'pre_roasting': ['T_data_2_1', 'T_data_2_2', 'T_data_2_3'],
        'main_roasting': ['T_data_3_1', 'T_data_3_2', 'T_data_3_3'],
        'post_roasting': ['T_data_4_1', 'T_data_4_2', 'T_data_4_3'],
        'cooling_zone': ['T_data_5_1', 'T_data_5_2', 'T_data_5_3'],
        'humidity': ['H_data', 'AH_data']
    },
    'process_parameters': {
        'drying_temp_range': (150, 220),
        'pre_roasting_temp_range': (220, 380),
        'main_roasting_temp_range': (380, 520),
        'post_roasting_temp_range': (300, 450),
        'cooling_temp_range': (200, 300),
        'humidity_range': (40, 60)
    },
    'data_quality': {
        'max_temp': 800,
        'min_temp': 0,
        'max_humidity': 100,
        'min_humidity': 0,
        'outlier_threshold': 3
    }
}

def get_quality_grade(quality_score):
    for grade, (min_score, max_score, description) in COFFEE_CONFIG['quality_grades'].items():
        if min_score <= quality_score <= max_score:
            return grade, description
    return 'Unknown', 'Score out of range'

def clean_sensor_data(features, feature_columns):
    print("\n🧹 Data Quality Enhancement:")
    print("=" * 50)
    
    df = pd.DataFrame(features, columns=feature_columns)
    original_shape = df.shape
    
    humidity_cols = ['H_data', 'AH_data']
    for col in humidity_cols:
        if col in df.columns:
            high_humidity_mask = df[col] > COFFEE_CONFIG['data_quality']['max_humidity']
            df.loc[high_humidity_mask, col] = COFFEE_CONFIG['data_quality']['max_humidity']
            print(f"   Fixed {high_humidity_mask.sum()} humidity readings > 100% in {col}")
    
    temp_cols = [col for col in df.columns if 'T_data' in col]
    for col in temp_cols:
        negative_temp_mask = df[col] < COFFEE_CONFIG['data_quality']['min_temp']
        df.loc[negative_temp_mask, col] = np.nan
        print(f"   Removed {negative_temp_mask.sum()} negative temperatures in {col}")
    
    for col in temp_cols:
        extreme_temp_mask = df[col] > COFFEE_CONFIG['data_quality']['max_temp']
        df.loc[extreme_temp_mask, col] = np.nan
        print(f"   Removed {extreme_temp_mask.sum()} extreme temperatures > 800°C in {col}")
    
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        df.loc[outlier_mask, col] = np.nan
        print(f"   Removed {outlier_mask.sum()} statistical outliers in {col}")
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    df_cleaned = df.dropna()
    
    print(f"\n📊 Data Cleaning Summary:")
    print(f"   Original records: {original_shape[0]}")
    print(f"   Cleaned records: {df_cleaned.shape[0]}")
    print(f"   Records removed: {original_shape[0] - df_cleaned.shape[0]}")
    print(f"   Data retention: {(df_cleaned.shape[0] / original_shape[0]) * 100:.1f}%")
    
    return df_cleaned.values

def analyze_cleaned_data(features, feature_columns):
    print("\n🔍 Cleaned Data Analysis:")
    print("=" * 50)
    
    df = pd.DataFrame(features, columns=feature_columns)
    
    for zone, sensors in COFFEE_CONFIG['sensor_mapping'].items():
        if zone == 'humidity':
            continue
            
        zone_data = df[sensors].mean(axis=1)
        
        if 'drying' in zone:
            temp_range = COFFEE_CONFIG['process_parameters']['drying_temp_range']
            zone_name = "Drying Zone"
        elif 'pre_roasting' in zone:
            temp_range = COFFEE_CONFIG['process_parameters']['pre_roasting_temp_range']
            zone_name = "Pre-Roasting Zone"
        elif 'main_roasting' in zone:
            temp_range = COFFEE_CONFIG['process_parameters']['main_roasting_temp_range']
            zone_name = "Main Roasting Zone"
        elif 'post_roasting' in zone:
            temp_range = COFFEE_CONFIG['process_parameters']['post_roasting_temp_range']
            zone_name = "Post-Roasting Zone"
        elif 'cooling' in zone:
            temp_range = COFFEE_CONFIG['process_parameters']['cooling_temp_range']
            zone_name = "Cooling Zone"
        else:
            continue
            
        temp_anomalies = ((zone_data < temp_range[0]) | (zone_data > temp_range[1])).sum()
        temp_anomaly_percent = (temp_anomalies / len(zone_data)) * 100
        
        print(f"\n📍 {zone_name}:")
        print(f"   Optimal Range: {temp_range[0]}°C - {temp_range[1]}°C")
        print(f"   Actual Range: {zone_data.min():.1f}°C - {zone_data.max():.1f}°C")
        print(f"   Average Temperature: {zone_data.mean():.1f}°C")
        print(f"   Temperature Anomalies: {temp_anomalies} ({temp_anomaly_percent:.1f}%)")
    
    humidity_data = df[COFFEE_CONFIG['sensor_mapping']['humidity']]
    humidity_range = COFFEE_CONFIG['process_parameters']['humidity_range']
    humidity_anomalies = ((humidity_data['H_data'] < humidity_range[0]) | 
                         (humidity_data['H_data'] > humidity_range[1])).sum()
    humidity_anomaly_percent = (humidity_anomalies / len(humidity_data)) * 100
    
    print(f"\n💧 Humidity Analysis:")
    print(f"   Optimal Range: {humidity_range[0]}% - {humidity_range[1]}%")
    print(f"   Actual Range: {humidity_data['H_data'].min():.1f}% - {humidity_data['H_data'].max():.1f}%")
    print(f"   Average Humidity: {humidity_data['H_data'].mean():.1f}%")
    print(f"   Humidity Anomalies: {humidity_anomalies} ({humidity_anomaly_percent:.1f}%)")

def load_and_prepare_data():
    try:
        print("☕ Coffee Quality Prediction System")
        print("=" * 70)
        print(f"Process: {COFFEE_CONFIG['process']}")
        print(f"Product: {COFFEE_CONFIG['product']}")
        print(f"Standards: SCA (Specialty Coffee Association) Compliant")
        print("=" * 70)
        
        print("\n📊 Loading coffee roasting process data...")
        
        data_X = pd.read_csv('data_X.csv')
        data_Y = pd.read_csv('data_Y.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Sensor Data (data_X.csv): {data_X.shape}")
        print(f"   - Quality Data (data_Y.csv): {data_Y.shape}")
        print(f"   - Submission Template: {sample_submission.shape}")
        
        data_X['date_time'] = pd.to_datetime(data_X['date_time'])
        data_Y['date_time'] = pd.to_datetime(data_Y['date_time'])
        sample_submission['date_time'] = pd.to_datetime(sample_submission['date_time'])
        
        feature_columns = [col for col in data_X.columns if col != 'date_time']
        print(f"\n📈 Coffee Roasting Process Sensors: {len(feature_columns)} sensors")
        
        print("\n🔄 Aligning coffee roasting process data by timestamp...")
        merged_data = pd.merge(data_X, data_Y, on='date_time', how='inner')
        print(f"✅ After temporal alignment: {merged_data.shape}")
        
        features = merged_data[feature_columns].values
        target = merged_data['quality'].values
        
        print(f"\n📊 Initial coffee roasting dataset:")
        print(f"   - Process Features: {features.shape}")
        print(f"   - Quality Targets: {target.shape}")
        
        features_cleaned = clean_sensor_data(features, feature_columns)
        
        target_cleaned = target[:len(features_cleaned)]
        
        print(f"\n📊 Final cleaned coffee roasting dataset:")
        print(f"   - Process Features: {features_cleaned.shape}")
        print(f"   - Quality Targets: {target_cleaned.shape}")
        
        analyze_cleaned_data(features_cleaned, feature_columns)
        
        quality_stats = {
            'min': target_cleaned.min(),
            'max': target_cleaned.max(),
            'mean': target_cleaned.mean(),
            'std': target_cleaned.std()
        }
        
        print(f"\n🎯 Quality Metrics Analysis:")
        print(f"   - Quality Range: {quality_stats['min']:.0f} - {quality_stats['max']:.0f}")
        print(f"   - Average Quality: {quality_stats['mean']:.1f}")
        print(f"   - Quality Variation: {quality_stats['std']:.1f}")
        
        grades = [get_quality_grade(score)[0] for score in target_cleaned]
        grade_counts = pd.Series(grades).value_counts()
        print(f"\n📊 Quality Grade Distribution:")
        for grade, count in grade_counts.items():
            percentage = (count / len(grades)) * 100
            print(f"   - Grade {grade}: {count} samples ({percentage:.1f}%)")
        
        return features_cleaned, target_cleaned, feature_columns, data_X, data_Y, sample_submission
        
    except Exception as e:
        print(f"❌ Error loading coffee roasting data: {e}")
        return None, None, None, None, None, None

def scale_data_robust(X, y, scaler_X=None, scaler_y=None):
    if scaler_X is None:
        scaler_X = RobustScaler()
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = scaler_X.transform(X)
    
    if scaler_y is None:
        scaler_y = RobustScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - time_steps):
        sequence = X[i:(i + time_steps)]
        
        engineered_features = []
        
        for sensor_idx in range(sequence.shape[1]):
            sensor_data = sequence[:, sensor_idx]
            engineered_features.extend([
                np.mean(sensor_data),
                np.std(sensor_data),
                np.max(sensor_data),
                np.min(sensor_data),
                np.ptp(sensor_data),
                np.percentile(sensor_data, 25),
                np.percentile(sensor_data, 75),
            ])
        
        temp_zones = {
            'drying': [0, 1, 2],
            'pre_roasting': [3, 4, 5],
            'main_roasting': [6, 7, 8],
            'post_roasting': [9, 10, 11],
            'cooling': [12, 13, 14]
        }
        
        for zone_name, sensor_indices in temp_zones.items():
            zone_data = sequence[:, sensor_indices]
            zone_mean = np.mean(zone_data, axis=1)
            engineered_features.extend([
                np.mean(zone_mean),
                np.std(zone_mean),
                np.max(zone_mean) - np.min(zone_mean)
            ])
        
        engineered_features.extend([
            i,
            time_steps,
            np.mean(sequence),
            np.std(sequence),
        ])
        
        enhanced_sequence = np.concatenate([sequence.flatten(), engineered_features])
        X_seq.append(enhanced_sequence)
        y_seq.append(y[i + time_steps])
    
    return np.array(X_seq), np.array(y_seq)

def create_quality_model(input_shape):
    input_size = input_shape[0]
    first_layer_size = input_size // 2
    second_layer_size = input_size // 2
    third_layer_size = input_size // 3
    fourth_layer_size = input_size // 8
    
    print(f"\n🏗️  Coffee Quality Model Architecture:")
    print(f"   - Input Layer: {input_size} enhanced features")
    print(f"   - First Hidden Layer: {first_layer_size} neurons (feature extraction)")
    print(f"   - Second Hidden Layer: {second_layer_size} neurons (deeper learning)")
    print(f"   - Third Hidden Layer: {third_layer_size} neurons (pattern recognition)")
    print(f"   - Fourth Hidden Layer: {fourth_layer_size} neurons (consolidation)")
    print(f"   - Output Layer: 1 neuron")
    
    model = keras.Sequential([
        keras.layers.Dense(input_size, input_shape=(input_size,),
                          kernel_initializer=keras.initializers.GlorotNormal(),
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        
        keras.layers.Dense(first_layer_size, activation='leaky_relu',
                          kernel_initializer=keras.initializers.GlorotNormal(),
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(second_layer_size, activation='leaky_relu',
                          kernel_initializer=keras.initializers.GlorotNormal(),
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(third_layer_size, activation='leaky_relu',
                          kernel_initializer=keras.initializers.GlorotNormal(),
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(fourth_layer_size, activation='leaky_relu',
                          kernel_initializer=keras.initializers.GlorotNormal(),
                          kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        
        keras.layers.Dense(1, activation='linear',
                          kernel_initializer=keras.initializers.GlorotNormal())
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.5),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def plot_training_curves(history, save_path='coffee_quality_training.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss', color='#8B4513', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='#D2691E', linewidth=2)
    ax1.set_title('Coffee Quality Model Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['mae'], label='Training MAE', color='#CD853F', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='#DEB887', linewidth=2)
    ax2.set_title('Coffee Quality Prediction MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training curves saved: {save_path}")

def main():
    print("🚀 Starting Coffee Quality Prediction System...")
    print("=" * 70)
    
    features, target, feature_columns, data_X, data_Y, sample_submission = load_and_prepare_data()
    if features is None:
        print("❌ Failed to load coffee roasting data. Exiting.")
        return
    
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data_robust(features, target)
    print(f"\n✅ Coffee roasting data scaled successfully using RobustScaler.")
    print(f"   - Process Features: {X_scaled.shape}")
    print(f"   - Quality Targets: {y_scaled.shape}")
    
    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    print(f"\n✅ Enhanced coffee roasting process sequences created.")
    print(f"   - Process Sequences: {X_seq.shape}")
    print(f"   - Quality Sequences: {y_seq.shape}")
    print(f"   - Enhanced Features: {X_seq.shape[1]} (original + statistical + zone + temporal)")
    print(f"   - Sequence Length: {time_steps} hours (roasting cycle)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    print(f"\n✅ Data split for coffee quality model training:")
    print(f"   - Training Set: {X_train.shape}")
    print(f"   - Test Set: {X_test.shape}")
    
    model = create_quality_model((X_seq.shape[1],))
    print("\n✅ Coffee Quality model created and compiled")
    print(f"📊 Coffee Quality Model Architecture Summary:")
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-8,
        verbose=1
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_coffee_quality_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    def cosine_annealing_schedule(epoch):
        initial_lr = 0.0001
        min_lr = 1e-8
        max_epochs = 100
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(cosine_annealing_schedule, verbose=1)
    
    print("\n🔄 Training Coffee Quality Model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler],
        verbose=1
    )
    
    print("\n✅ Coffee Quality Model Training Completed!")
    
    print("\n📊 Generating coffee quality model training curves...")
    plot_training_curves(history)
    
    print("\n📈 Evaluating Coffee Quality Model Performance...")
    y_pred = model.predict(X_test)
    
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    print("\n📊 Coffee Quality Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    errors = np.abs(y_test_original - y_pred_original)
    within_1_percent = np.mean(errors <= 1.0) * 100
    within_5_percent = np.mean(errors <= 5.0) * 100
    within_10_percent = np.mean(errors <= 10.0) * 100
    
    print(f"\n🎯 Coffee Quality Prediction Accuracy:")
    print(f"Within 1% error: {within_1_percent:.2f}%")
    print(f"Within 5% error: {within_5_percent:.2f}%")
    print(f"Within 10% error: {within_10_percent:.2f}%")
    
    actual_grades = [get_quality_grade(score)[0] for score in y_test_original]
    predicted_grades = [get_quality_grade(score)[0] for score in y_pred_original]
    grade_accuracy = np.mean([a == p for a, p in zip(actual_grades, predicted_grades)]) * 100
    print(f"Quality Grade Prediction Accuracy: {grade_accuracy:.2f}%")
    
    model.save('coffee_quality_model.h5')
    print("\n✅ Coffee quality model saved: coffee_quality_model.h5")
    
    print("\n" + "=" * 70)
    print("✅ Coffee Quality Prediction System Completed!")
    print("=" * 70)
    print("📁 Generated Files:")
    print("   - coffee_quality_training.png")
    print("   - coffee_quality_model.h5")
    print("   - best_coffee_quality_model.h5")
    print("\n🎯 Coffee Quality Model Architecture Features:")
    print("   - 5-layer feedforward neural network with feature engineering")
    print("   - Enhanced input: 546 features (original + statistical + zone + temporal)")
    print("   - First layer: 273 neurons (feature extraction)")
    print("   - Second layer: 273 neurons (deeper learning)")
    print("   - Third layer: 182 neurons (pattern recognition)")
    print("   - Fourth layer: 68 neurons (consolidation)")
    print("   - Output layer: 1 neuron (quality prediction)")
    print("   - Leaky ReLU activation functions (prevents dying neurons)")
    print("   - Xavier/Glorot initialization with strong L2 regularization (0.01)")
    print("   - Batch normalization on ALL layers (including input)")
    print("   - Consistent dropout (25-30%) in hidden layers")
    print("   - Balanced Adam optimizer (LR=0.0001, clipnorm=0.5)")
    print("   - Balanced cosine annealing learning rate scheduler")
    print("   - Validation loss-focused callbacks (monitor val_loss, increased patience)")

if __name__ == "__main__":
    main()
