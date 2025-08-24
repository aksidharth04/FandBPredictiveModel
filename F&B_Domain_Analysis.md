# ☕ Coffee Bean Roasting Process Anomaly Prediction - Domain Analysis

## 🎯 **Identified F&B Process: Industrial Coffee Bean Roasting**

Based on the sensor data analysis, this project models an **Industrial Coffee Bean Roasting Process** with the following characteristics:

### **Process Overview**
- **Product**: Commercial coffee bean roasting (specialty coffee production)
- **Scale**: Large-scale roasting facility (1,000-10,000 kg/day)
- **Process Type**: Batch roasting with multiple temperature zones
- **Quality Focus**: Coffee bean quality, flavor development, and consistency

## 📊 **Sensor Data Analysis & Coffee Roasting Relevance**

### **Temperature Sensors (15 sensors)**
```
T_data_1_1, T_data_1_2, T_data_1_3 → Drying Zone Temperatures (91-446°C)
T_data_2_1, T_data_2_2, T_data_2_3 → Pre-Roasting Zone Temperatures (105-637°C)  
T_data_3_1, T_data_3_2, T_data_3_3 → Main Roasting Zone Temperatures (45-1172°C)
T_data_4_1, T_data_4_2, T_data_4_3 → Post-Roasting Zone Temperatures (17-666°C)
T_data_5_1, T_data_5_2, T_data_5_3 → Cooling Zone Temperatures (114-465°C)
```

**Coffee Roasting Relevance:**
- **Drying Zone**: Removes moisture from green beans (200-250°C)
- **Pre-Roasting**: Bean temperature rise and color change (300-400°C)
- **Main Roasting**: Critical flavor development and chemical reactions (400-600°C)
- **Post-Roasting**: Flavor stabilization and development (300-400°C)
- **Cooling Zone**: Rapid cooling to stop roasting process (200-250°C)

### **Humidity Sensors (2 sensors)**
```
H_data   → Relative Humidity (%)
AH_data  → Absolute Humidity (g/m³)
```

**Coffee Roasting Relevance:**
- **Bean Moisture**: Affects roasting time and flavor development
- **Roasting Environment**: Humidity impacts heat transfer and bean expansion
- **Cooling Process**: Prevents condensation and maintains bean quality
- **Storage Conditions**: Maintains optimal moisture content for shelf life

## 🏭 **Industrial Coffee Roasting Process Steps**

### **1. Green Bean Preparation**
- **Bean Variety**: Arabica, Robusta, or blends
- **Moisture Content**: 10-12% for optimal roasting
- **Bean Size**: Uniform sizing for consistent roasting
- **Quality Assessment**: Defect removal and grading

### **2. Drying Phase**
- **Equipment**: Drum roaster with temperature control
- **Parameters**: 200-250°C, 3-5 minutes
- **Quality Indicators**: Bean color change, moisture reduction
- **Sensor Relevance**: T_data_1_1, T_data_1_2, T_data_1_3, H_data

### **3. Pre-Roasting Phase**
- **Equipment**: Main roasting drum
- **Parameters**: 300-400°C, 5-8 minutes
- **Quality Indicators**: First crack, bean expansion, color development
- **Sensor Relevance**: T_data_2_1, T_data_2_2, T_data_2_3, H_data, AH_data

### **4. Main Roasting Phase**
- **Equipment**: High-temperature roasting drum
- **Parameters**: 400-600°C, 8-12 minutes
- **Quality Indicators**: Second crack, oil development, flavor compounds
- **Sensor Relevance**: T_data_3_1, T_data_3_2, T_data_3_3

### **5. Post-Roasting Phase**
- **Equipment**: Cooling and stabilization zone
- **Parameters**: 300-400°C, 2-3 minutes
- **Quality Indicators**: Flavor stabilization, color uniformity
- **Sensor Relevance**: T_data_4_1, T_data_4_2, T_data_4_3

### **6. Cooling & Packaging**
- **Equipment**: Air cooling system, packaging lines
- **Parameters**: 200-250°C, rapid cooling
- **Quality Indicators**: Final moisture, shelf-life, appearance
- **Sensor Relevance**: T_data_5_1, T_data_5_2, T_data_5_3, H_data, AH_data

## 📈 **Quality Metrics Definition**

### **Primary Quality Indicators (Quality Score: 221-505)**

Based on research and industry standards, the quality score represents a **composite quality index**:

#### **1. Roast Quality (40%)**
- **Roast Level**: Light, Medium, Medium-Dark, Dark (target: consistent level)
- **Color Uniformity**: Even roast color across batch
- **Bean Expansion**: Proper bean size increase (target: 15-20%)
- **Surface Oil**: Appropriate oil development for roast level

#### **2. Flavor Quality (30%)**
- **Acidity**: Bright, balanced acidity levels
- **Body**: Full, rich mouthfeel
- **Aroma**: Complex, appealing fragrance
- **Taste Balance**: Harmony of flavors (sweet, bitter, sour)

#### **3. Technical Quality (20%)**
- **Moisture Content**: Final moisture (target: 1-3%)
- **Bean Integrity**: Minimal breakage and defects
- **Shelf Life**: Oxidation resistance and freshness retention
- **Consistency**: Batch-to-batch uniformity

#### **4. Process Efficiency (10%)**
- **Energy Efficiency**: Optimal heat utilization
- **Waste Reduction**: Minimizing defective beans
- **Production Speed**: Throughput optimization
- **Cost Control**: Efficient resource utilization

### **Quality Score Interpretation**
```
Quality Range    | Grade    | Description
-----------------|----------|------------------
450-505          | A+       | Excellent quality
400-449          | A        | Good quality
350-399          | B        | Acceptable quality
300-349          | C        | Below standard
221-299          | D        | Poor quality (reject)
```

## 🔬 **Research-Based Quality Standards**

### **Industry Standards (SCA - Specialty Coffee Association)**
- **Roast Level**: Consistent color and development
- **Moisture Content**: 1-3% for optimal shelf life
- **Bean Expansion**: 15-20% volume increase
- **Defect Rate**: <5% broken or defective beans

### **Process Control Parameters**
- **Drying Temperature**: 200-250°C for moisture removal
- **Pre-Roasting Temperature**: 300-400°C for first crack
- **Main Roasting Temperature**: 400-600°C for flavor development
- **Cooling Temperature**: 200-250°C for rapid cooling
- **Relative Humidity**: 40-60% for optimal roasting environment

## 🎯 **Anomaly Detection Focus Areas**

### **1. Temperature Anomalies**
- **Drying Zone**: Deviations from 200-250°C range affecting moisture removal
- **Pre-Roasting**: Temperature fluctuations affecting first crack timing
- **Main Roasting**: Uneven heating causing inconsistent flavor development
- **Cooling**: Rapid cooling preventing proper flavor stabilization

### **2. Humidity Anomalies**
- **Low Humidity**: Excessive bean drying, poor flavor development
- **High Humidity**: Poor heat transfer, inconsistent roasting
- **Humidity Fluctuations**: Inconsistent bean quality and flavor

### **3. Process Timing Anomalies**
- **Drying Time**: Over/under-drying affecting bean structure
- **Roasting Time**: Insufficient or excessive flavor development
- **Cooling Time**: Improper cooling affecting final quality

### **4. Quality Prediction Targets**
- **Early Warning**: Detect quality issues 2-4 hours before they occur
- **Process Optimization**: Identify optimal parameter ranges
- **Cost Reduction**: Minimize waste and rework
- **Consistency**: Maintain quality across batches

## 📚 **References & Industry Standards**

1. **Specialty Coffee Association (SCA)**: Roasting Standards and Protocols
2. **Coffee Quality Institute (CQI)**: Quality Assessment Guidelines
3. **International Coffee Organization (ICO)**: Industry Standards
4. **European Coffee Federation**: Quality Control Protocols
5. **Food Safety Modernization Act (FSMA)**: Preventive Controls

## 🏭 **Industrial Application Context**

### **Commercial Coffee Roasting Operations**
- **Scale**: 1,000-10,000 kg per day
- **Equipment**: Industrial drum roasters, cooling systems
- **Automation**: PLC-controlled temperature profiles
- **Quality Control**: Real-time monitoring and adjustment

### **Economic Impact**
- **Quality Losses**: 5-15% of production due to quality issues
- **Energy Costs**: 20-30% of production costs
- **Waste Reduction**: 10-20% improvement potential
- **Shelf Life**: 6-12 months extension through optimal processing

This domain analysis provides the foundation for a comprehensive coffee roasting process anomaly prediction system specifically tailored to industrial coffee bean roasting operations.
