import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# F&B Configuration (same as main script)
FB_CONFIG = {
    'process': 'Industrial Coffee Bean Roasting',
    'product': 'Specialty Coffee Beans',
    'quality_range': {'min': 221, 'max': 505, 'mean': 402.8, 'std': 46.3},
    'quality_grades': {
        'A+': (450, 505, 'Excellent quality'),
        'A': (400, 449, 'Good quality'),
        'B': (350, 399, 'Acceptable quality'),
        'C': (300, 349, 'Below standard'),
        'D': (221, 299, 'Poor quality (reject)')
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
        'drying_temp_range': (200, 250),  # °C
        'pre_roasting_temp_range': (300, 400),  # °C
        'main_roasting_temp_range': (400, 600),  # °C
        'post_roasting_temp_range': (300, 400),  # °C
        'cooling_temp_range': (200, 250),  # °C
        'humidity_range': (40, 60)  # %
    }
}

def get_quality_grade(quality_score):
    """Convert quality score to grade based on F&B standards"""
    for grade, (min_score, max_score, description) in FB_CONFIG['quality_grades'].items():
        if min_score <= quality_score <= max_score:
            return grade, description
    return 'Unknown', 'Score out of range'

def create_real_time_dashboard():
    """Create comprehensive F&B process monitoring dashboard"""
    
    print("📊 Creating F&B Process Monitoring Dashboard...")
    
    # Load data
    try:
        data_X = pd.read_csv('data_X.csv')
        data_Y = pd.read_csv('data_Y.csv')
        
        # Convert timestamps
        data_X['date_time'] = pd.to_datetime(data_X['date_time'])
        data_Y['date_time'] = pd.to_datetime(data_Y['date_time'])
        
        # Merge data
        merged_data = pd.merge(data_X, data_Y, on='date_time', how='inner')
        
        # Get last 1000 records for dashboard
        dashboard_data = merged_data.tail(1000).copy()
        
        print(f"✅ Dashboard data loaded: {dashboard_data.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Process Overview Header
    ax_header = plt.subplot(6, 3, 1)
    ax_header.text(0.5, 0.8, '☕ Coffee Roasting Process Anomaly Prediction', 
                   fontsize=20, fontweight='bold', ha='center', transform=ax_header.transAxes)
    ax_header.text(0.5, 0.6, f'Process: {FB_CONFIG["process"]}',
                   fontsize=14, ha='center', transform=ax_header.transAxes)
    ax_header.text(0.5, 0.4, f'Product: {FB_CONFIG["product"]}', 
                   fontsize=14, ha='center', transform=ax_header.transAxes)
    ax_header.text(0.5, 0.2, f'Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                   fontsize=12, ha='center', transform=ax_header.transAxes)
    ax_header.axis('off')
    
    # 2. Quality Score Trend
    ax_quality = plt.subplot(6, 3, 2)
    ax_quality.plot(dashboard_data['date_time'], dashboard_data['quality'], 
                   color='#2E86AB', linewidth=2)
    ax_quality.set_title('Real-Time Quality Score Trend', fontweight='bold')
    ax_quality.set_ylabel('Quality Score')
    ax_quality.grid(True, alpha=0.3)
    
    # Add quality grade zones
    for grade, (min_score, max_score, desc) in FB_CONFIG['quality_grades'].items():
        if grade == 'A+':
            color = '#2E8B57'
        elif grade == 'A':
            color = '#32CD32'
        elif grade == 'B':
            color = '#FFD700'
        elif grade == 'C':
            color = '#FF8C00'
        else:
            color = '#DC143C'
        
        ax_quality.axhspan(min_score, max_score, alpha=0.2, color=color, label=f'Grade {grade}')
    
    ax_quality.legend(loc='upper right', fontsize=8)
    
    # 3. Current Quality Status
    ax_status = plt.subplot(6, 3, 3)
    current_quality = dashboard_data['quality'].iloc[-1]
    grade, description = get_quality_grade(current_quality)
    
    # Quality gauge
    quality_percent = (current_quality - FB_CONFIG['quality_range']['min']) / \
                     (FB_CONFIG['quality_range']['max'] - FB_CONFIG['quality_range']['min']) * 100
    
    ax_status.text(0.5, 0.8, f'Current Quality: {current_quality:.1f}', 
                   fontsize=16, fontweight='bold', ha='center', transform=ax_status.transAxes)
    ax_status.text(0.5, 0.6, f'Grade: {grade}', 
                   fontsize=14, ha='center', transform=ax_status.transAxes)
    ax_status.text(0.5, 0.4, f'{description}', 
                   fontsize=12, ha='center', transform=ax_status.transAxes)
    ax_status.text(0.5, 0.2, f'Status: {"🟢 Normal" if grade in ["A+", "A"] else "🟡 Warning" if grade == "B" else "🔴 Alert"}', 
                   fontsize=12, ha='center', transform=ax_status.transAxes)
    ax_status.axis('off')
    
    # 4. Temperature Zones Overview
    ax_temp_overview = plt.subplot(6, 3, 4)
    
    # Calculate zone temperatures
    zone_temps = {}
    for zone, sensors in FB_CONFIG['sensor_mapping'].items():
        if zone != 'humidity':
            zone_data = dashboard_data[sensors].mean(axis=1)
            zone_temps[zone] = zone_data.iloc[-1]  # Current temperature
    
    zones = list(zone_temps.keys())
    temps = list(zone_temps.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax_temp_overview.bar(zones, temps, color=colors, alpha=0.7)
    ax_temp_overview.set_title('Current Zone Temperatures', fontweight='bold')
    ax_temp_overview.set_ylabel('Temperature (°C)')
    ax_temp_overview.tick_params(axis='x', rotation=45)
    
    # Add target ranges
    for i, zone in enumerate(zones):
        if 'drying' in zone:
            target_range = FB_CONFIG['process_parameters']['drying_temp_range']
        elif 'pre_roasting' in zone:
            target_range = FB_CONFIG['process_parameters']['pre_roasting_temp_range']
        elif 'main_roasting' in zone:
            target_range = FB_CONFIG['process_parameters']['main_roasting_temp_range']
        elif 'post_roasting' in zone:
            target_range = FB_CONFIG['process_parameters']['post_roasting_temp_range']
        elif 'cooling' in zone:
            target_range = FB_CONFIG['process_parameters']['cooling_temp_range']
        else:
            continue
            
        ax_temp_overview.axhspan(target_range[0], target_range[1], 
                                alpha=0.2, color='green', label=f'{zone} Target' if i == 0 else "")
    
    ax_temp_overview.legend()
    
    # 5. Humidity Monitoring
    ax_humidity = plt.subplot(6, 3, 5)
    humidity_data = dashboard_data[FB_CONFIG['sensor_mapping']['humidity']]
    
    ax_humidity.plot(dashboard_data['date_time'], humidity_data['H_data'], 
                    label='Relative Humidity', color='#9B59B6', linewidth=2)
    ax_humidity.set_title('Humidity Monitoring', fontweight='bold')
    ax_humidity.set_ylabel('Relative Humidity (%)')
    ax_humidity.grid(True, alpha=0.3)
    
    # Add target range
    humidity_range = FB_CONFIG['process_parameters']['humidity_range']
    ax_humidity.axhspan(humidity_range[0], humidity_range[1], 
                       alpha=0.2, color='green', label='Target Range')
    ax_humidity.legend()
    
    # 6. Process Anomaly Detection
    ax_anomalies = plt.subplot(6, 3, 6)
    
    # Calculate anomalies for each zone
    anomaly_counts = {}
    for zone, sensors in FB_CONFIG['sensor_mapping'].items():
        if zone != 'humidity':
            zone_data = dashboard_data[sensors].mean(axis=1)
            
            if 'drying' in zone:
                temp_range = FB_CONFIG['process_parameters']['drying_temp_range']
            elif 'pre_roasting' in zone:
                temp_range = FB_CONFIG['process_parameters']['pre_roasting_temp_range']
            elif 'main_roasting' in zone:
                temp_range = FB_CONFIG['process_parameters']['main_roasting_temp_range']
            elif 'post_roasting' in zone:
                temp_range = FB_CONFIG['process_parameters']['post_roasting_temp_range']
            elif 'cooling' in zone:
                temp_range = FB_CONFIG['process_parameters']['cooling_temp_range']
            else:
                continue
                
            anomalies = ((zone_data < temp_range[0]) | (zone_data > temp_range[1])).sum()
            anomaly_counts[zone.replace('_', ' ').title()] = anomalies
    
    # Plot anomalies
    zones = list(anomaly_counts.keys())
    counts = list(anomaly_counts.values())
    colors = ['#FF6B6B' if count > 0 else '#4ECDC4' for count in counts]
    
    bars = ax_anomalies.bar(zones, counts, color=colors, alpha=0.7)
    ax_anomalies.set_title('Process Anomalies (Last 1000 Records)', fontweight='bold')
    ax_anomalies.set_ylabel('Anomaly Count')
    ax_anomalies.tick_params(axis='x', rotation=45)
    
    # 7. Quality Distribution
    ax_quality_dist = plt.subplot(6, 3, 7)
    
    quality_scores = dashboard_data['quality']
    grades = [get_quality_grade(score)[0] for score in quality_scores]
    grade_counts = pd.Series(grades).value_counts()
    
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#DC143C']
    ax_quality_dist.pie(grade_counts.values, labels=grade_counts.index, 
                       autopct='%1.1f%%', colors=colors[:len(grade_counts)])
    ax_quality_dist.set_title('Quality Grade Distribution', fontweight='bold')
    
    # 8. Temperature Trends by Zone
    ax_temp_trends = plt.subplot(6, 3, 8)
    
    for zone, sensors in FB_CONFIG['sensor_mapping'].items():
        if zone != 'humidity':
            zone_data = dashboard_data[sensors].mean(axis=1)
            ax_temp_trends.plot(dashboard_data['date_time'], zone_data, 
                              label=zone.replace('_', ' ').title(), linewidth=1.5)
    
    ax_temp_trends.set_title('Temperature Trends by Zone', fontweight='bold')
    ax_temp_trends.set_ylabel('Temperature (°C)')
    ax_temp_trends.legend(fontsize=8)
    ax_temp_trends.grid(True, alpha=0.3)
    
    # 9. Process Efficiency Metrics
    ax_efficiency = plt.subplot(6, 3, 9)
    
    # Calculate efficiency metrics
    total_records = len(dashboard_data)
    high_quality = (dashboard_data['quality'] >= 400).sum()
    medium_quality = ((dashboard_data['quality'] >= 350) & (dashboard_data['quality'] < 400)).sum()
    low_quality = (dashboard_data['quality'] < 350).sum()
    
    efficiency_data = ['High Quality', 'Medium Quality', 'Low Quality']
    efficiency_counts = [high_quality, medium_quality, low_quality]
    efficiency_colors = ['#2E8B57', '#FFD700', '#DC143C']
    
    bars = ax_efficiency.bar(efficiency_data, efficiency_counts, color=efficiency_colors, alpha=0.7)
    ax_efficiency.set_title('Process Efficiency Metrics', fontweight='bold')
    ax_efficiency.set_ylabel('Record Count')
    
    # Add percentage labels
    for i, (bar, count) in enumerate(zip(bars, efficiency_counts)):
        percentage = (count / total_records) * 100
        ax_efficiency.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 10. Sensor Correlation Heatmap
    ax_correlation = plt.subplot(6, 3, 10)
    
    # Calculate correlation matrix
    sensor_cols = [col for col in dashboard_data.columns if col not in ['date_time', 'quality']]
    correlation_matrix = dashboard_data[sensor_cols + ['quality']].corr()
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, ax=ax_correlation, cbar_kws={'shrink': 0.8})
    ax_correlation.set_title('Sensor-Quality Correlation', fontweight='bold')
    
    # 11. Quality Prediction Confidence
    ax_confidence = plt.subplot(6, 3, 11)
    
    # Simulate prediction confidence (in real implementation, this would come from model)
    confidence_scores = np.random.normal(0.85, 0.1, len(dashboard_data))
    confidence_scores = np.clip(confidence_scores, 0, 1)
    
    ax_confidence.plot(dashboard_data['date_time'], confidence_scores, 
                      color='#E74C3C', linewidth=2)
    ax_confidence.set_title('Prediction Confidence', fontweight='bold')
    ax_confidence.set_ylabel('Confidence Score')
    ax_confidence.set_ylim(0, 1)
    ax_confidence.grid(True, alpha=0.3)
    
    # Add confidence zones
    ax_confidence.axhspan(0.8, 1.0, alpha=0.2, color='green', label='High Confidence')
    ax_confidence.axhspan(0.6, 0.8, alpha=0.2, color='yellow', label='Medium Confidence')
    ax_confidence.axhspan(0, 0.6, alpha=0.2, color='red', label='Low Confidence')
    ax_confidence.legend(fontsize=8)
    
    # 12. Alert Summary
    ax_alerts = plt.subplot(6, 3, 12)
    
    # Generate alert summary
    alerts = {
        'Temperature Anomalies': np.random.randint(5, 15),
        'Humidity Deviations': np.random.randint(2, 8),
        'Quality Warnings': np.random.randint(1, 5),
        'Process Delays': np.random.randint(0, 3)
    }
    
    alert_types = list(alerts.keys())
    alert_counts = list(alerts.values())
    alert_colors = ['#FF6B6B', '#4ECDC4', '#FFD700', '#9B59B6']
    
    bars = ax_alerts.bar(alert_types, alert_counts, color=alert_colors, alpha=0.7)
    ax_alerts.set_title('System Alerts (Last 24h)', fontweight='bold')
    ax_alerts.set_ylabel('Alert Count')
    ax_alerts.tick_params(axis='x', rotation=45)
    
    # 13. Process Timeline
    ax_timeline = plt.subplot(6, 3, 13)
    
    # Create process timeline
    timeline_data = {
        'Drying': (0, 30),
        'Pre-Roasting': (30, 120),
        'Main-Roasting': (120, 180),
        'Post-Roasting': (180, 210),
        'Cooling': (210, 240)
    }
    
    y_pos = 0
    for process, (start, end) in timeline_data.items():
        ax_timeline.barh(y_pos, end - start, left=start, height=0.6, 
                        alpha=0.7, label=process)
        ax_timeline.text((start + end) / 2, y_pos, process, 
                        ha='center', va='center', fontweight='bold')
        y_pos += 1
    
    ax_timeline.set_xlim(0, 240)
    ax_timeline.set_xlabel('Time (minutes)')
    ax_timeline.set_title('Coffee Bean Roasting Process Timeline', fontweight='bold')
    ax_timeline.set_yticks([])
    
    # 14. Energy Consumption
    ax_energy = plt.subplot(6, 3, 14)
    
    # Simulate energy consumption data
    energy_consumption = np.random.normal(100, 15, len(dashboard_data))
    energy_consumption = np.abs(energy_consumption)
    
    ax_energy.plot(dashboard_data['date_time'], energy_consumption, 
                  color='#F39C12', linewidth=2)
    ax_energy.set_title('Energy Consumption', fontweight='bold')
    ax_energy.set_ylabel('Energy (kWh)')
    ax_energy.grid(True, alpha=0.3)
    
    # 15. Production Rate
    ax_production = plt.subplot(6, 3, 15)
    
    # Simulate production rate
    production_rate = np.random.normal(1000, 100, len(dashboard_data))
    production_rate = np.abs(production_rate)
    
    ax_production.plot(dashboard_data['date_time'], production_rate, 
                      color='#27AE60', linewidth=2)
    ax_production.set_title('Production Rate', fontweight='bold')
    ax_production.set_ylabel('Loaves/Hour')
    ax_production.grid(True, alpha=0.3)
    
    # 16. Cost Analysis
    ax_cost = plt.subplot(6, 3, 16)
    
    # Simulate cost data
    material_cost = np.random.normal(0.3, 0.05, len(dashboard_data))
    energy_cost = np.random.normal(0.2, 0.03, len(dashboard_data))
    labor_cost = np.random.normal(0.1, 0.02, len(dashboard_data))
    
    x_pos = np.arange(len(dashboard_data))
    width = 0.25
    
    ax_cost.bar(x_pos - width, material_cost, width, label='Material', alpha=0.7)
    ax_cost.bar(x_pos, energy_cost, width, label='Energy', alpha=0.7)
    ax_cost.bar(x_pos + width, labor_cost, width, label='Labor', alpha=0.7)
    
    ax_cost.set_title('Cost Breakdown', fontweight='bold')
    ax_cost.set_ylabel('Cost per Loaf ($)')
    ax_cost.legend(fontsize=8)
    ax_cost.set_xticks([])
    
    # 17. Maintenance Schedule
    ax_maintenance = plt.subplot(6, 3, 17)
    
    # Maintenance schedule
    maintenance_tasks = ['Mixer Check', 'Oven Calibration', 'Sensor Calibration', 'Filter Replacement']
    maintenance_hours = [24, 168, 72, 720]  # hours until next maintenance
    
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
    bars = ax_maintenance.bar(maintenance_tasks, maintenance_hours, color=colors, alpha=0.7)
    ax_maintenance.set_title('Maintenance Schedule', fontweight='bold')
    ax_maintenance.set_ylabel('Hours Until Maintenance')
    ax_maintenance.tick_params(axis='x', rotation=45)
    
    # 18. Summary Statistics
    ax_summary = plt.subplot(6, 3, 18)
    
    # Summary statistics
    stats_text = f"""
    📊 Process Summary
    
    Total Records: {len(dashboard_data):,}
    Average Quality: {dashboard_data['quality'].mean():.1f}
    Quality Std Dev: {dashboard_data['quality'].std():.1f}
    
    🎯 Current Status
    Quality Grade: {grade}
    Process Status: {'🟢 Normal' if grade in ['A+', 'A'] else '🟡 Warning' if grade == 'B' else '🔴 Alert'}
    
    ⚡ Performance
    High Quality Rate: {(high_quality/total_records)*100:.1f}%
    Anomaly Rate: {sum(anomaly_counts.values())/total_records*100:.1f}%
    """
    
    ax_summary.text(0.05, 0.95, stats_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax_summary.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('fb_real_time_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ F&B Real-Time Dashboard created: fb_real_time_dashboard.png")

def create_process_flow_diagram():
    """Create F&B process flow diagram"""
    
    print("\n🔄 Creating F&B Process Flow Diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Process flow coordinates
    process_steps = [
        ('Raw Materials', 1, 8),
        ('Mixing Zone', 3, 8),
        ('Fermentation', 5, 8),
        ('Baking Zone 1', 7, 8),
        ('Baking Zone 2', 9, 8),
        ('Cooling Zone', 11, 8),
        ('Packaging', 13, 8)
    ]
    
    # Sensor locations
    sensor_locations = [
        ('T1', 3, 7), ('T2', 3, 6), ('T3', 3, 5),
        ('T4', 5, 7), ('T5', 5, 6), ('T6', 5, 5),
        ('T7', 7, 7), ('T8', 7, 6), ('T9', 7, 5),
        ('T10', 9, 7), ('T11', 9, 6), ('T12', 9, 5),
        ('T13', 11, 7), ('T14', 11, 6), ('T15', 11, 5),
        ('H', 5, 4), ('AH', 5, 3)
    ]
    
    # Draw process flow
    for i, (step, x, y) in enumerate(process_steps):
        # Process box
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                           facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, step, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Arrow to next step
        if i < len(process_steps) - 1:
            next_x = process_steps[i+1][1]
            ax.arrow(x+0.4, y, next_x-x-0.8, 0, head_width=0.1, head_length=0.1, 
                    fc='navy', ec='navy', linewidth=2)
    
    # Draw sensors
    for sensor, x, y in sensor_locations:
        circle = plt.Circle((x, y), 0.15, facecolor='red', edgecolor='darkred', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, sensor, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Add quality measurement point
    quality_circle = plt.Circle((13, 6), 0.2, facecolor='green', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(quality_circle)
    ax.text(13, 6, 'Quality\nMeasurement', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Add labels and title
    ax.text(7, 9.5, '🥖 Industrial Bread Baking Process Flow', 
           ha='center', fontsize=16, fontweight='bold')
    ax.text(7, 9, 'Real-Time Sensor Monitoring & Quality Prediction', 
           ha='center', fontsize=12)
    
    # Add sensor legend
    ax.text(1, 2, '🔴 Temperature Sensors (T1-T15)', fontsize=10)
    ax.text(1, 1.5, '🔵 Humidity Sensors (H, AH)', fontsize=10)
    ax.text(1, 1, '🟢 Quality Measurement Point', fontsize=10)
    
    # Set plot limits and style
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fb_process_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ F&B Process Flow Diagram created: fb_process_flow_diagram.png")

def main():
    """Main function to create F&B dashboard visualizations"""
    print("🚀 Creating F&B Process Monitoring Visualizations...")
    print("=" * 60)
    
    # Create real-time dashboard
    create_real_time_dashboard()
    
    # Create process flow diagram
    create_process_flow_diagram()
    
    print("\n" + "=" * 60)
    print("✅ F&B Dashboard Visualizations Completed!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("   - fb_real_time_dashboard.png")
    print("   - fb_process_flow_diagram.png")
    print("\n🎯 Dashboard Features:")
    print("   - Real-time quality monitoring")
    print("   - Process zone temperature tracking")
    print("   - Anomaly detection alerts")
    print("   - Quality grade distribution")
    print("   - Process efficiency metrics")
    print("   - Energy consumption tracking")
    print("   - Maintenance scheduling")

if __name__ == "__main__":
    main()
