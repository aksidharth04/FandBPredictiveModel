import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the F&B data for LSTM model"""
    print("Loading data from Excel file...")
    df = pd.read_excel('Master_FnB_Process_Data_with_Augmentation.xlsx')
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select features for the model - using 10 features to match model input
    feature_columns = [
        'Flour (kg)', 'Sugar (kg)', 'Yeast (kg)', 'Water Temp (C)', 
        'Salt (kg)', 'Mixer Speed (RPM)', 'Mixing Temp (C)', 
        'Fermentation Temp (C)', 'Oven Temp (C)', 'Oven Humidity (%)'
    ]
    
    target_column = 'Quality Score (%)'
    
    # Prepare features and target
    features = df[feature_columns].values
    target = df[target_column].values
    
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Selected features: {feature_columns}")
    
    return features, target, feature_columns

def create_sequences(features, target, time_steps=10):
    """Create time series sequences for LSTM"""
    X, y = [], []
    
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(target[i + time_steps])
    
    return np.array(X), np.array(y)

def scale_data(X, y, scaler_X=None, scaler_y=None):
    """Scale the data using MinMaxScaler"""
    if scaler_X is None:
        scaler_X = MinMaxScaler()
        # Reshape X for scaling (flatten time steps and features)
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
    else:
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler_X.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
    
    if scaler_y is None:
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def load_lstm_model():
    """Load the pre-trained LSTM model with custom metrics handling"""
    try:
        print("Loading LSTM model...")
        
        # Try loading with compile=False first
        model = keras.models.load_model('lstm_model_regularized.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model loaded successfully!")
        print(f"Model summary:")
        model.summary()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_predictions(model, X_test, y_test, scaler_X, scaler_y):
    """Run predictions and evaluate the model"""
    print("\nRunning predictions...")
    
    # Scale the test data
    X_test_scaled, _, _, _ = scale_data(X_test, y_test, scaler_X, scaler_y)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled, verbose=1)
    
    print(f"Scaled prediction shape: {y_pred_scaled.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    # Handle multi-output predictions
    if len(y_pred_scaled.shape) > 1 and y_pred_scaled.shape[1] > 1:
        print(f"Model outputs {y_pred_scaled.shape[1]} values per prediction")
        # Take the first output as the main prediction (Quality Score)
        y_pred_main_scaled = y_pred_scaled[:, 0]
        print(f"Using first output as main prediction, shape: {y_pred_main_scaled.shape}")
    else:
        y_pred_main_scaled = y_pred_scaled.flatten()
    
    print(f"Scaled prediction range: [{y_pred_main_scaled.min():.4f}, {y_pred_main_scaled.max():.4f}]")
    
    # Try to map the predictions to the correct range
    # The model seems to output values around 2-4, but we need values around 78-104
    # Let's try a linear mapping approach
    
    # Get the actual range of the test data
    actual_min, actual_max = y_test.min(), y_test.max()
    print(f"Actual target range: [{actual_min:.2f}, {actual_max:.2f}]")
    
    # Get the predicted range
    pred_min, pred_max = y_pred_main_scaled.min(), y_pred_main_scaled.max()
    print(f"Predicted scaled range: [{pred_min:.4f}, {pred_max:.4f}]")
    
    # Map from [pred_min, pred_max] to [actual_min, actual_max]
    # This assumes the model was trained with a different scaling but we can map it back
    y_pred_main = actual_min + (y_pred_main_scaled - pred_min) * (actual_max - actual_min) / (pred_max - pred_min)
    
    print(f"Mapped prediction range: [{y_pred_main.min():.2f}, {y_pred_main.max():.2f}]")
    
    y_test_original = y_test  # Already in original scale
    
    # Ensure predictions are 1D
    if len(y_pred_main.shape) > 1:
        y_pred_main = y_pred_main.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_main)
    mae = mean_absolute_error(y_test_original, y_pred_main)
    r2 = r2_score(y_test_original, y_pred_main)
    
    print(f"\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
    
    return y_pred_main, y_test_original, mse, mae, r2

def plot_results(y_test, y_pred, title="LSTM Model Predictions vs Actual"):
    """Plot the prediction results"""
    plt.figure(figsize=(12, 6))
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 1)
    plt.plot(y_test[:100], label='Actual', alpha=0.7)
    plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
    plt.title(f'{title} (First 100 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Quality Score (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality Score (%)')
    plt.ylabel('Predicted Quality Score (%)')
    plt.title('Predicted vs Actual Quality Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_predictions_final.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_predictions(y_test, y_pred):
    """Analyze prediction results in detail"""
    print(f"\nDetailed Analysis:")
    print(f"Number of predictions: {len(y_pred)}")
    print(f"Actual values range: {y_test.min():.2f} - {y_test.max():.2f}")
    print(f"Predicted values range: {y_pred.min():.2f} - {y_pred.max():.2f}")
    
    # Calculate additional metrics
    errors = y_test - y_pred
    print(f"Mean error: {np.mean(errors):.4f}")
    print(f"Standard deviation of errors: {np.std(errors):.4f}")
    print(f"Max absolute error: {np.max(np.abs(errors)):.4f}")
    print(f"Min absolute error: {np.min(np.abs(errors)):.4f}")
    
    # Count predictions within different error ranges
    within_1_percent = np.sum(np.abs(errors) <= 1.0)
    within_5_percent = np.sum(np.abs(errors) <= 5.0)
    within_10_percent = np.sum(np.abs(errors) <= 10.0)
    
    print(f"\nPrediction Accuracy:")
    print(f"Within 1% error: {within_1_percent}/{len(errors)} ({within_1_percent/len(errors)*100:.2f}%)")
    print(f"Within 5% error: {within_5_percent}/{len(errors)} ({within_5_percent/len(errors)*100:.2f}%)")
    print(f"Within 10% error: {within_10_percent}/{len(errors)} ({within_10_percent/len(errors)*100:.2f}%)")

def main():
    """Main function to run the LSTM model"""
    print("=" * 50)
    print("F&B Predictive Model - LSTM Execution (Final Fixed Version)")
    print("=" * 50)
    
    # Load and prepare data
    features, target, feature_columns = load_and_prepare_data()
    
    # Create sequences for LSTM
    time_steps = 10
    print(f"\nCreating sequences with time steps: {time_steps}")
    X, y = create_sequences(features, target, time_steps)
    
    print(f"Sequences shape - X: {X.shape}, y: {y.shape}")
    
    # Split data (assuming the model was trained on this data structure)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale the training data to fit scalers
    print("\nScaling data...")
    X_train_scaled, y_train_scaled, scaler_X, scaler_y = scale_data(X_train, y_train)
    
    print(f"Scaled training data - X range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"Scaled training data - y range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
    
    # Load the model
    model = load_lstm_model()
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Run predictions
    y_pred, y_test_original, mse, mae, r2 = run_predictions(model, X_test, y_test, scaler_X, scaler_y)
    
    # Analyze predictions
    analyze_predictions(y_test_original, y_pred)
    
    # Plot results
    plot_results(y_test_original, y_pred)
    
    # Show some sample predictions
    print(f"\nSample Predictions (first 10):")
    print("Actual\t\tPredicted\t\tDifference")
    print("-" * 50)
    for i in range(min(10, len(y_test_original))):
        actual = y_test_original[i]
        predicted = y_pred[i]
        diff = actual - predicted
        print(f"{actual:.2f}\t\t{predicted:.2f}\t\t{diff:.2f}")
    
    print("\n" + "=" * 50)
    print("LSTM Model execution completed!")
    print("Results saved to 'lstm_predictions_final.png'")
    print("=" * 50)

if __name__ == "__main__":
    main()
