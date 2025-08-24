# ☕ Coffee Roasting Industry Research - Optimal Parameters & Standards

## 🏭 **Industrial Coffee Roasting Machine Analysis**

### **Types of Commercial Roasters**

#### **1. Drum Roasters (Most Common)**
- **Capacity**: 1-500 kg per batch
- **Temperature Range**: 150-600°C
- **Control**: PLC-based temperature profiling
- **Applications**: Specialty coffee, commercial roasting

#### **2. Fluid Bed Roasters**
- **Capacity**: 50-1000 kg per batch
- **Temperature Range**: 200-500°C
- **Control**: Air temperature and flow rate
- **Applications**: High-volume commercial roasting

#### **3. Hot Air Roasters**
- **Capacity**: 100-2000 kg per batch
- **Temperature Range**: 180-550°C
- **Control**: Air temperature and velocity
- **Applications**: Large-scale industrial roasting

## 📊 **Industry-Standard Temperature Profiles**

### **SCA (Specialty Coffee Association) Standards**

#### **Light Roast Profile**
```
Phase          | Temperature | Duration | Key Events
---------------|-------------|----------|------------
Drying         | 150-200°C   | 3-5 min  | Moisture removal
Yellowing      | 200-250°C   | 2-3 min  | Bean color change
First Crack    | 380-420°C   | 1-2 min  | First audible crack
Development    | 420-450°C   | 1-3 min  | Flavor development
Cooling        | 200-250°C   | 2-3 min  | Rapid cooling
```

#### **Medium Roast Profile**
```
Phase          | Temperature | Duration | Key Events
---------------|-------------|----------|------------
Drying         | 150-200°C   | 3-5 min  | Moisture removal
Yellowing      | 200-280°C   | 3-4 min  | Bean expansion
First Crack    | 380-420°C   | 2-3 min  | First audible crack
Development    | 420-480°C   | 2-4 min  | Flavor development
Second Crack   | 480-520°C   | 1-2 min  | Second audible crack
Cooling        | 200-250°C   | 2-3 min  | Rapid cooling
```

#### **Dark Roast Profile**
```
Phase          | Temperature | Duration | Key Events
---------------|-------------|----------|------------
Drying         | 150-200°C   | 3-5 min  | Moisture removal
Yellowing      | 200-300°C   | 4-5 min  | Bean expansion
First Crack    | 380-420°C   | 3-4 min  | First audible crack
Development    | 420-520°C   | 4-6 min  | Flavor development
Second Crack   | 480-550°C   | 2-3 min  | Second audible crack
Cooling        | 200-250°C   | 2-3 min  | Rapid cooling
```

## 🔬 **Optimal Temperature Ranges by Zone**

### **Zone 1: Drying Zone (T_data_1_1, T_data_1_2, T_data_1_3)**
- **Optimal Range**: 150-220°C
- **Purpose**: Remove moisture from green beans
- **Duration**: 3-5 minutes
- **Key Indicators**: Bean color change from green to yellow
- **Current Data**: 91.3-446.3°C (avg: 250.2°C)
- **Analysis**: Higher than optimal, may cause over-drying

### **Zone 2: Pre-Roasting Zone (T_data_2_1, T_data_2_2, T_data_2_3)**
- **Optimal Range**: 220-380°C
- **Purpose**: Bean expansion and color development
- **Duration**: 4-6 minutes
- **Key Indicators**: Bean expansion, color change
- **Current Data**: -105.7-637.0°C (avg: 349.8°C)
- **Analysis**: Wide range, includes negative values (sensor issues)

### **Zone 3: Main Roasting Zone (T_data_3_1, T_data_3_2, T_data_3_3)**
- **Optimal Range**: 380-520°C
- **Purpose**: Critical flavor development and cracking
- **Duration**: 6-10 minutes
- **Key Indicators**: First and second crack
- **Current Data**: 45.0-1172.0°C (avg: 501.2°C)
- **Analysis**: Extremely high temperatures, may cause over-roasting

### **Zone 4: Post-Roasting Zone (T_data_4_1, T_data_4_2, T_data_4_3)**
- **Optimal Range**: 300-450°C
- **Purpose**: Flavor stabilization and development
- **Duration**: 2-4 minutes
- **Key Indicators**: Flavor compound development
- **Current Data**: 16.7-666.0°C (avg: 349.6°C)
- **Analysis**: Good average, but wide range

### **Zone 5: Cooling Zone (T_data_5_1, T_data_5_2, T_data_5_3)**
- **Optimal Range**: 200-300°C
- **Purpose**: Rapid cooling to stop roasting process
- **Duration**: 2-3 minutes
- **Key Indicators**: Temperature stabilization
- **Current Data**: 114.3-464.7°C (avg: 249.7°C)
- **Analysis**: Good average, but upper range too high

## 💧 **Humidity Control Standards**

### **Optimal Humidity Ranges**
- **Roasting Environment**: 40-60% RH
- **Green Bean Storage**: 10-12% moisture content
- **Roasted Bean Storage**: 1-3% moisture content
- **Current Data**: 141.5-207.8% RH (avg: 174.7%)
- **Analysis**: Extremely high humidity, indicates sensor calibration issues

### **Humidity Impact on Roasting**
- **Low Humidity (<30%)**: Excessive bean drying, poor flavor development
- **Optimal Humidity (40-60%)**: Balanced heat transfer, consistent roasting
- **High Humidity (>70%)**: Poor heat transfer, inconsistent roasting
- **Very High Humidity (>100%)**: Sensor malfunction or calibration error

## 🎯 **Quality Metrics & Standards**

### **SCA Quality Assessment Criteria**

#### **1. Roast Level Consistency (40%)**
- **Light Roast**: Agtron #95-85
- **Medium Roast**: Agtron #85-65
- **Medium-Dark Roast**: Agtron #65-45
- **Dark Roast**: Agtron #45-25

#### **2. Bean Development (30%)**
- **Color Uniformity**: Even roast across batch
- **Bean Expansion**: 15-20% volume increase
- **Surface Oil**: Appropriate for roast level
- **Bean Integrity**: Minimal breakage

#### **3. Flavor Profile (20%)**
- **Acidity**: Bright, balanced
- **Body**: Full, rich mouthfeel
- **Aroma**: Complex, appealing
- **Taste Balance**: Harmony of flavors

#### **4. Technical Quality (10%)**
- **Moisture Content**: 1-3%
- **Defect Rate**: <5% broken beans
- **Shelf Life**: 6-12 months
- **Consistency**: Batch-to-batch uniformity

## 🏭 **Industrial Roasting Machine Specifications**

### **Commercial Drum Roaster (Typical)**
```
Specification        | Value
--------------------|------------------
Capacity            | 100-500 kg/batch
Power Consumption   | 50-200 kW
Temperature Range   | 150-600°C
Control System      | PLC with PID control
Data Logging        | Real-time monitoring
Cooling System      | Air-cooled or water-cooled
Batch Time          | 12-20 minutes
```

### **Temperature Control Systems**
- **PID Controllers**: Precise temperature control
- **Multi-zone Heating**: Independent zone control
- **Real-time Monitoring**: Continuous temperature tracking
- **Data Logging**: Historical temperature profiles
- **Alarm Systems**: Temperature deviation alerts

## 📈 **Process Optimization Parameters**

### **Optimal Roasting Curves**

#### **Light Roast (City Roast)**
```
Time (min) | Temperature (°C) | Event
-----------|------------------|------------------
0-3        | 150-200         | Drying
3-6        | 200-280         | Yellowing
6-8        | 280-380         | Development
8-10       | 380-420         | First Crack
10-12      | 420-450         | Light Development
12-14      | 450-200         | Cooling
```

#### **Medium Roast (Full City Roast)**
```
Time (min) | Temperature (°C) | Event
-----------|------------------|------------------
0-3        | 150-200         | Drying
3-6        | 200-300         | Yellowing
6-9        | 300-380         | Development
9-11       | 380-420         | First Crack
11-14      | 420-480         | Development
14-16      | 480-200         | Cooling
```

#### **Dark Roast (Vienna Roast)**
```
Time (min) | Temperature (°C) | Event
-----------|------------------|------------------
0-3        | 150-200         | Drying
3-6        | 200-320         | Yellowing
6-10       | 320-380         | Development
10-12      | 380-420         | First Crack
12-16      | 420-520         | Development
16-18      | 520-550         | Second Crack
18-20      | 550-200         | Cooling
```

## 🔧 **Equipment Maintenance Standards**

### **Regular Maintenance Schedule**
- **Daily**: Temperature sensor calibration check
- **Weekly**: Drum cleaning and inspection
- **Monthly**: Full system calibration
- **Quarterly**: Major component inspection
- **Annually**: Complete system overhaul

### **Critical Components**
- **Temperature Sensors**: Calibrate every 30 days
- **Heating Elements**: Inspect every 90 days
- **Cooling System**: Clean every 60 days
- **Control System**: Update software every 6 months
- **Data Logging**: Backup data weekly

## 📊 **Data Analysis Recommendations**

### **Sensor Calibration Issues**
Based on the current data analysis:
1. **Zone 2**: Negative temperatures indicate sensor malfunction
2. **Zone 3**: Extremely high temperatures (1172°C) suggest sensor error
3. **Humidity**: Values >100% indicate sensor calibration issues
4. **Temperature Ranges**: Much wider than industry standards

### **Recommended Actions**
1. **Sensor Calibration**: Recalibrate all temperature sensors
2. **Humidity Sensor**: Replace or recalibrate humidity sensors
3. **Data Validation**: Implement data quality checks
4. **Process Optimization**: Adjust temperature profiles to industry standards

## 🎯 **Quality Prediction Model Optimization**

### **Updated Temperature Ranges for Model**
```python
'process_parameters': {
    'drying_temp_range': (150, 220),      # °C - Optimal drying
    'pre_roasting_temp_range': (220, 380), # °C - Pre-crack development
    'main_roasting_temp_range': (380, 520), # °C - Crack and development
    'post_roasting_temp_range': (300, 450), # °C - Flavor stabilization
    'cooling_temp_range': (200, 300),     # °C - Rapid cooling
    'humidity_range': (40, 60)            # % - Optimal roasting environment
}
```

### **Anomaly Detection Thresholds**
- **Temperature Deviation**: ±20°C from optimal range
- **Humidity Deviation**: ±10% from optimal range
- **Process Time Deviation**: ±2 minutes from standard
- **Quality Score Threshold**: <350 (Grade C or below)

This research provides the foundation for optimizing the coffee roasting anomaly prediction system with industry-standard parameters and best practices.
