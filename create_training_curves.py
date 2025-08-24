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
        # Load the data
        df = pd.read_excel('Master_FnB_Process_Data_with_Augmentation.xlsx')
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Select features (10 features to match the model)
        feature_columns = [
            'Flour (kg)', 'Sugar (kg)', 'Yeast (kg)', 'Water Temp (C)',
            'Salt (kg)', 'Mixer Speed (RPM)', 'Mixing Temp (C)',
            'Fermentation Temp (C)', 'Oven Temp (C)', 'Oven Humidity (%)'
        ]
        target_column = 'Quality Score (%)'
        
        # Check if all columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Use available features
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
            return None, None, None
        
        # Extract features and target
        features = df[feature_columns].values
        target = df[target_column].values
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_indices]
        target = target[valid_indices]
        
        print(f"After cleaning: Features shape: {features.shape}, Target shape: {target.shape}")
        print(f"Feature columns: {feature_columns}")
        
        return features, target, feature_columns
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

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

def create_sequences(X, y, time_steps=10):
    """Create time series sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def create_lstm_model(input_shape, output_size=1):
    """Create LSTM model with improved stability"""
    model = keras.Sequential([
        keras.layers.LSTM(50, input_shape=input_shape, return_sequences=False,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='glorot_uniform'),
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
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # MAE curves
    ax2.plot(history.history['mae'], label='Training MAE', color='green', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    ax2.set_title('LSTM Training and Validation MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_loss_curves(history):
    """Analyze the loss curves and provide insights"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss convergence analysis
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    ax1.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(val_loss, label='Validation Loss', color='red', linewidth=2)
    ax1.axhline(y=min(val_loss), color='green', linestyle='--', alpha=0.7, 
                label=f'Best Val Loss: {min(val_loss):.4f}')
    ax1.set_title('Loss Convergence Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Overfitting analysis
    overfitting_ratio = [val/train if train > 0 else 1 for train, val in zip(train_loss, val_loss)]
    ax2.plot(overfitting_ratio, color='purple', linewidth=2)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Overfitting Line')
    ax2.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    ax2.set_title('Overfitting Analysis\n(Val Loss / Train Loss)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss stability analysis
    train_loss_diff = np.diff(train_loss)
    val_loss_diff = np.diff(val_loss)
    
    ax3.plot(train_loss_diff, label='Training Loss Change', color='blue', alpha=0.7)
    ax3.plot(val_loss_diff, label='Validation Loss Change', color='red', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('Loss Stability Analysis\n(Loss Change per Epoch)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Change')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to generate training curves"""
    print("🚀 Starting LSTM Training Curves Generation...")
    
    # Load and prepare data
    features, target, feature_columns = load_and_prepare_data()
    if features is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Scale the data
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data_safely(features, target)
    print(f"✅ Data scaled successfully. X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")
    
    # Create sequences
    time_steps = 10
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
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    print("🔄 Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("✅ Training completed!")
    
    # Plot training curves
    print("📊 Generating training curves...")
    plot_training_curves(history)
    
    # Analyze loss curves
    print("📈 Analyzing loss patterns...")
    analyze_loss_curves(history)
    
    # Evaluate model
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
    
    print("\n✅ Training curves generation completed!")
    print("📁 Generated files:")
    print("   - lstm_training_curves.png")
    print("   - lstm_loss_analysis.png")

if __name__ == "__main__":
    main()
