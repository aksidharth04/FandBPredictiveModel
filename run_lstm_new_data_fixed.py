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

def load_and_prepare_data():
    """Load and prepare data for LSTM training"""
    try:
        print("📊 Loading data files...")
        
        # Load the data files
        data_X = pd.read_csv('data_X.csv')
        data_Y = pd.read_csv('data_Y.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
        
        print(f"✅ Data loaded successfully:")
        print(f"   - data_X.csv: {data_X.shape}")
        print(f"   - data_Y.csv: {data_Y.shape}")
        print(f"   - sample_submission.csv: {sample_submission.shape}")
        
        # Convert date_time to datetime
        data_X['date_time'] = pd.to_datetime(data_X['date_time'])
        data_Y['date_time'] = pd.to_datetime(data_Y['date_time'])
        sample_submission['date_time'] = pd.to_datetime(sample_submission['date_time'])
        
        # Get feature columns (exclude date_time)
        feature_columns = [col for col in data_X.columns if col != 'date_time']
        print(f"📈 Features: {len(feature_columns)} columns")
        print(f"   Feature columns: {feature_columns}")
        
        # Align data by date_time
        print("🔄 Aligning data by date_time...")
        
        # Merge data_X and data_Y on date_time
        merged_data = pd.merge(data_X, data_Y, on='date_time', how='inner')
        print(f"✅ After merging: {merged_data.shape}")
        
        # Extract features and target
        features = merged_data[feature_columns].values
        target = merged_data['quality'].values
        
        print(f"📊 Final data shapes:")
        print(f"   - Features: {features.shape}")
        print(f"   - Target: {target.shape}")
        
        # Check for NaN values
        nan_features = np.isnan(features).sum()
        nan_target = np.isnan(target).sum()
        print(f"🔍 Data quality check:")
        print(f"   - NaN in features: {nan_features}")
        print(f"   - NaN in target: {nan_target}")
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_indices]
        target = target[valid_indices]
        
        print(f"✅ After cleaning:")
        print(f"   - Features: {features.shape}")
        print(f"   - Target: {target.shape}")
        
        return features, target, feature_columns, data_X, data_Y, sample_submission
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
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
    """Create time series sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def create_lstm_model(input_shape, output_size=1):
    """Create LSTM model optimized for the new data"""
    model = keras.Sequential([
        # First LSTM layer with more units for complex patterns
        keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='glorot_uniform'),
        keras.layers.Dropout(0.3),
        
        # Second LSTM layer
        keras.layers.LSTM(64, return_sequences=False,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='glorot_uniform'),
        keras.layers.Dropout(0.3),
        
        # Dense layers
        keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_size, kernel_initializer='glorot_uniform')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def plot_training_curves(history):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('LSTM Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # MAE curves
    ax2.plot(history.history['mae'], label='Training MAE', color='green', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    ax2.set_title('LSTM Training and Validation MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_new_data_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Quality"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality', fontsize=12)
    plt.ylabel('Predicted Quality', fontsize=12)
    plt.title(f'{title}\nScatter Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot time series
    plt.subplot(1, 2, 2)
    plt.plot(y_true[:100], label='Actual', color='blue', linewidth=2)
    plt.plot(y_pred[:100], label='Predicted', color='red', linewidth=2)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Quality', fontsize=12)
    plt.title(f'{title}\nTime Series (First 100 points)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_new_data_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_submission_predictions(model, scaler_X, scaler_y, sample_submission, data_X, feature_columns):
    """Generate predictions for submission format"""
    print("🎯 Generating submission predictions...")
    
    # Get the dates from sample submission
    submission_dates = sample_submission['date_time'].values
    
    # Find corresponding features for submission dates
    submission_features = []
    for date in submission_dates:
        # Find the closest date in data_X
        date_diff = abs(data_X['date_time'] - pd.to_datetime(date))
        closest_idx = date_diff.idxmin()
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
    else:
        # If not enough data, use a default value
        sample_submission['quality'] = 400  # Default quality value
    
    # Save submission file
    sample_submission.to_csv('submission_predictions.csv', index=False)
    print(f"✅ Submission file saved: submission_predictions.csv")
    
    return sample_submission

def main():
    """Main function to run LSTM model on new data"""
    print("🚀 Starting LSTM Model for New Data...")
    
    # Load and prepare data
    features, target, feature_columns, data_X, data_Y, sample_submission = load_and_prepare_data()
    if features is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Scale the data
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data_safely(features, target)
    print(f"✅ Data scaled successfully. X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")
    
    # Create sequences (24 time steps = 24 hours)
    time_steps = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    print(f"✅ Sequences created. X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    print(f"✅ Data split. Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create and compile model
    model = create_lstm_model((time_steps, X_scaled.shape[1]))
    print("✅ LSTM model created and compiled")
    print(f"📊 Model summary:")
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
    print("🔄 Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("✅ Training completed!")
    
    # Plot training curves
    print("📊 Generating training curves...")
    plot_training_curves(history)
    
    # Evaluate model
    print("📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    print("\n📊 Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Calculate accuracy within different thresholds
    errors = np.abs(y_test_original - y_pred_original)
    within_1_percent = np.mean(errors <= 1.0) * 100
    within_5_percent = np.mean(errors <= 5.0) * 100
    within_10_percent = np.mean(errors <= 10.0) * 100
    
    print(f"\n🎯 Prediction Accuracy:")
    print(f"Within 1% error: {within_1_percent:.2f}%")
    print(f"Within 5% error: {within_5_percent:.2f}%")
    print(f"Within 10% error: {within_10_percent:.2f}%")
    
    # Plot predictions
    print("📊 Generating prediction plots...")
    plot_predictions(y_test_original, y_pred_original, "LSTM Quality Prediction")
    
    # Generate submission predictions
    submission_result = generate_submission_predictions(
        model, scaler_X, scaler_y, sample_submission, data_X, feature_columns
    )
    
    # Save the trained model
    model.save('lstm_new_data_model.h5')
    print("✅ Model saved: lstm_new_data_model.h5")
    
    print("\n✅ LSTM model for new data completed!")
    print("📁 Generated files:")
    print("   - lstm_new_data_training_curves.png")
    print("   - lstm_new_data_predictions.png")
    print("   - submission_predictions.csv")
    print("   - lstm_new_data_model.h5")

if __name__ == "__main__":
    main()
