# Coffee Quality Prediction System

**Machine learning system for predicting coffee bean quality during industrial roasting processes**

## Overview

This system uses deep learning to predict coffee quality scores based on real-time sensor data from industrial roasting equipment. The model analyzes temperature readings from multiple zones and humidity data to forecast final product quality.

## Machine Learning Architecture

### Model Type
- **Neural Network**: 5-layer feedforward network with feature engineering
- **Input**: 546 enhanced features (original + statistical + zone + temporal)
- **Output**: Single quality score prediction

### Feature Engineering
- **Statistical Features**: Mean, std, min, max, range, percentiles for each sensor
- **Zone Analysis**: Temperature correlations across roasting zones
- **Temporal Features**: Time position, sequence length, overall statistics

### Network Architecture
- **Input Layer**: 546 neurons (enhanced features)
- **Hidden Layer 1**: 273 neurons (feature extraction)
- **Hidden Layer 2**: 273 neurons (deeper learning)
- **Hidden Layer 3**: 182 neurons (pattern recognition)
- **Hidden Layer 4**: 68 neurons (consolidation)
- **Output Layer**: 1 neuron (quality prediction)

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.0001, clipnorm: 0.5)
- **Loss Function**: Mean Squared Error
- **Regularization**: L2 regularization (0.01) on all layers
- **Activation**: Leaky ReLU (prevents dying neurons)
- **Normalization**: Batch normalization on all layers
- **Dropout**: 25-30% in hidden layers

## Data Processing

### Sensor Data
- **Temperature Zones**: 5 zones (drying, pre-roasting, main roasting, post-roasting, cooling)
- **Sensors per Zone**: 3 temperature sensors each
- **Humidity Sensors**: 2 sensors (relative and absolute humidity)
- **Total Sensors**: 17 sensors

### Data Cleaning
- **Outlier Removal**: IQR method for statistical outliers
- **Calibration Fixes**: Humidity capped at 100%, negative temperatures removed
- **Missing Values**: Forward fill and backward fill
- **Data Retention**: Typically 85-95% after cleaning

### Scaling
- **Method**: RobustScaler (handles outliers better than MinMaxScaler)
- **Features**: Robust scaling for all sensor data
- **Target**: Robust scaling for quality scores

## Quality Grading System

Based on Specialty Coffee Association (SCA) standards:

- **A+ (450-505)**: Excellent quality - Premium specialty coffee
- **A (400-449)**: Good quality - Commercial specialty coffee  
- **B (350-399)**: Acceptable quality - Standard commercial coffee
- **C (300-349)**: Below standard - Requires process adjustment
- **D (221-299)**: Poor quality - Reject batch

## Performance Metrics

### Model Evaluation
- **Mean Squared Error (MSE)**: Primary loss metric
- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Squared Error (RMSE)**: Standard deviation of errors
- **R² Score**: Coefficient of determination

### Accuracy Benchmarks
- **Within 1% error**: Target accuracy
- **Within 5% error**: Acceptable accuracy
- **Within 10% error**: Minimum accuracy
- **Grade Prediction**: Accuracy of quality grade classification

## Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy, matplotlib

### Installation
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
python coffee_quality_model.py
```

### Input Data Format
- **data_X.csv**: Sensor readings with timestamp
- **data_Y.csv**: Quality scores with timestamp
- **sample_submission.csv**: Prediction template

### Output
- **coffee_quality_model.h5**: Trained model
- **best_coffee_quality_model.h5**: Best checkpoint
- **coffee_quality_training.png**: Training curves

## Process Parameters

### Temperature Ranges (SCA Guidelines)
- **Drying Zone**: 150-220°C
- **Pre-Roasting**: 220-380°C
- **Main Roasting**: 380-520°C
- **Post-Roasting**: 300-450°C
- **Cooling Zone**: 200-300°C

### Humidity Control
- **Optimal Range**: 40-60%
- **Maximum**: 100% (sensor calibration limit)
- **Minimum**: 0%

## Technical Specifications

### Data Quality Thresholds
- **Max Temperature**: 800°C (realistic limit)
- **Min Temperature**: 0°C
- **Outlier Detection**: 3 standard deviations
- **Sequence Length**: 24 time steps (roasting cycles)

### Model Training
- **Epochs**: 100 maximum
- **Batch Size**: 128
- **Validation Split**: 20%
- **Early Stopping**: Patience of 25 epochs
- **Learning Rate**: Cosine annealing scheduler

## Dashboard Interface

The system includes a React-based dashboard for real-time monitoring:

- **Real-time Quality Tracking**: Live quality score updates
- **Temperature Monitoring**: Zone-by-zone temperature analysis
- **Process Efficiency**: Quality breakdown and efficiency metrics
- **Alert System**: Real-time anomaly detection
- **Model Performance**: Training metrics and validation curves

## Files

- **coffee_quality_model.py**: Main ML model implementation
- **honeywell_dashboard.py**: Additional dashboard functionality
- **src/**: React dashboard components
- **requirements.txt**: Python dependencies
- **package.json**: Node.js dependencies

## Author

**Adicherikandi Sidharth**
- GitHub: [@aksidharth04](https://github.com/aksidharth04)
- Email: aksidharthm10@gmail.com

## License

MIT License
