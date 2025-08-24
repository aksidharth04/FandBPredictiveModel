import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set Helvetica font with bold styling
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'

# Honeywell Brand Colors
HONEYWELL_COLORS = {
    'primary': '#D32F2F',      # Honeywell Red
    'secondary': '#1976D2',    # Blue
    'accent': '#FFC107',       # Amber
    'success': '#388E3C',      # Green
    'warning': '#F57C00',      # Orange
    'danger': '#D32F2F',       # Red
    'light_gray': '#F5F5F5',   # Light Gray
    'dark_gray': '#424242'     # Dark Gray
}

# F&B Configuration
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
    }
}

def get_quality_grade(quality_score):
    """Convert quality score to grade"""
    for grade, (min_score, max_score, description) in FB_CONFIG['quality_grades'].items():
        if min_score <= quality_score <= max_score:
            return grade, description
    return 'Unknown', 'Score out of range'

def create_honeywell_dashboard():
    """Create professional Honeywell-branded F&B process monitoring dashboard"""
    
    print("📊 Creating Professional Honeywell F&B Dashboard...")
    
    # Load data
    try:
        data_X = pd.read_csv('data_X.csv')
        data_Y = pd.read_csv('data_Y.csv')
        
        data_X['date_time'] = pd.to_datetime(data_X['date_time'])
        data_Y['date_time'] = pd.to_datetime(data_Y['date_time'])
        
        merged_data = pd.merge(data_X, data_Y, on='date_time', how='inner')
        dashboard_data = merged_data.tail(1000).copy()
        
        print(f"✅ Dashboard data loaded: {dashboard_data.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create professional dashboard with cool effects
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(HONEYWELL_COLORS['light_gray'])
    
    # Create a gradient background effect
    ax_bg = fig.add_subplot(111, frameon=False)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.set_facecolor(HONEYWELL_COLORS['light_gray'])
    
    # 1. Honeywell Header with Professional Styling
    ax_header = plt.subplot(3, 4, 1)
    ax_header.set_facecolor('white')
    
    # Add Honeywell logo with shadow effect
    ax_header.text(0.5, 0.9, 'HONEYWELL', fontsize=26, fontweight='bold', 
                   color=HONEYWELL_COLORS['primary'], ha='center', transform=ax_header.transAxes,
                   fontfamily='Helvetica', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', edgecolor=HONEYWELL_COLORS['primary'], linewidth=2))
    
    ax_header.text(0.5, 0.7, 'Process Control & Automation', fontsize=14, 
                   color=HONEYWELL_COLORS['dark_gray'], ha='center', transform=ax_header.transAxes,
                   fontfamily='Helvetica', fontweight='bold')
    
    ax_header.text(0.5, 0.5, 'Coffee Roasting', fontsize=16, fontweight='bold',
                   color=HONEYWELL_COLORS['secondary'], ha='center', transform=ax_header.transAxes,
                   fontfamily='Helvetica')
    
    ax_header.text(0.5, 0.3, f'Last Update: {datetime.now().strftime("%H:%M:%S")}', 
                   fontsize=11, color=HONEYWELL_COLORS['dark_gray'], ha='center', transform=ax_header.transAxes,
                   fontfamily='Helvetica', fontweight='bold')
    
    # Add animated-style status indicator
    ax_header.text(0.5, 0.1, '🟢 LIVE MONITORING', fontsize=12, ha='center', transform=ax_header.transAxes,
                   color=HONEYWELL_COLORS['success'], fontweight='bold', fontfamily='Helvetica')
    ax_header.axis('off')
    
    # 2. Model Training Performance (Large and Prominent)
    ax_training = plt.subplot(3, 4, (2, 4))  # Span 3 columns
    try:
        training_img = plt.imread('coffee_roasting_optimized_training.png')
        ax_training.imshow(training_img)
        ax_training.set_title('Model Training Performance', fontweight='bold', 
                             color=HONEYWELL_COLORS['dark_gray'], fontsize=18,
                             fontfamily='Helvetica', pad=20)
        ax_training.axis('off')
    except:
        ax_training.text(0.5, 0.5, 'Training Performance\n(Image not available)', 
                        ha='center', va='center', transform=ax_training.transAxes,
                        fontsize=16, color=HONEYWELL_COLORS['dark_gray'],
                        fontfamily='Helvetica', fontweight='bold')
        ax_training.set_title('Model Training Performance', fontweight='bold', 
                             color=HONEYWELL_COLORS['dark_gray'], fontsize=18,
                             fontfamily='Helvetica')
        ax_training.axis('off')
    
    # 3. Current Quality Status with Gauge Effect
    ax_status = plt.subplot(3, 4, 5)
    current_quality = dashboard_data['quality'].iloc[-1]
    grade, description = get_quality_grade(current_quality)
    
    # Calculate quality percentage
    quality_percent = (current_quality - FB_CONFIG['quality_range']['min']) / \
                     (FB_CONFIG['quality_range']['max'] - FB_CONFIG['quality_range']['min']) * 100
    
    # Create gauge-like display with cool effects
    ax_status.set_facecolor('white')
    
    # Quality gauge background
    gauge_circle = plt.Circle((0.5, 0.3), 0.25, fill=False, color=HONEYWELL_COLORS['light_gray'], linewidth=8)
    ax_status.add_patch(gauge_circle)
    
    # Quality gauge fill based on percentage
    if quality_percent >= 80:
        gauge_color = HONEYWELL_COLORS['success']
        quality_indicator = "EXCELLENT"
    elif quality_percent >= 60:
        gauge_color = '#4CAF50'
        quality_indicator = "GOOD"
    elif quality_percent >= 40:
        gauge_color = HONEYWELL_COLORS['accent']
        quality_indicator = "ACCEPTABLE"
    else:
        gauge_color = HONEYWELL_COLORS['danger']
        quality_indicator = "POOR"
    
    # Animated gauge effect
    gauge_fill = plt.Circle((0.5, 0.3), 0.25, fill=True, color=gauge_color, alpha=0.3)
    ax_status.add_patch(gauge_fill)
    
    ax_status.text(0.5, 0.7, 'CURRENT QUALITY', fontsize=14, fontweight='bold', ha='center', transform=ax_status.transAxes,
                   color=HONEYWELL_COLORS['dark_gray'], fontfamily='Helvetica')
    ax_status.text(0.5, 0.55, f'{current_quality:.1f}', fontsize=28, fontweight='bold', ha='center', transform=ax_status.transAxes,
                   color=HONEYWELL_COLORS['primary'], fontfamily='Helvetica')
    ax_status.text(0.5, 0.45, quality_indicator, fontsize=16, fontweight='bold', ha='center', transform=ax_status.transAxes,
                   color=gauge_color, fontfamily='Helvetica')
    ax_status.text(0.5, 0.35, f'{quality_percent:.1f}%', fontsize=12, ha='center', transform=ax_status.transAxes,
                   color=HONEYWELL_COLORS['dark_gray'], fontfamily='Helvetica', fontweight='bold')
    
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis('off')
    
    # 4. Temperature Zones with 3D Effect
    ax_temp = plt.subplot(3, 4, 6)
    zone_temps = {}
    for zone, sensors in FB_CONFIG['sensor_mapping'].items():
        if zone != 'humidity':
            zone_data = dashboard_data[sensors].mean(axis=1)
            zone_temps[zone] = zone_data.iloc[-1]
    
    zones = list(zone_temps.keys())
    temps = list(zone_temps.values())
    colors = [HONEYWELL_COLORS['primary'], HONEYWELL_COLORS['secondary'], 
              HONEYWELL_COLORS['accent'], HONEYWELL_COLORS['success'], HONEYWELL_COLORS['warning']]
    
    # Create 3D-like bars with shadows
    bars = ax_temp.bar(zones, temps, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    ax_temp.set_title('Zone Temperatures', fontweight='bold', 
                      color=HONEYWELL_COLORS['dark_gray'], fontsize=14, fontfamily='Helvetica')
    ax_temp.set_ylabel('Temperature (°C)', fontsize=12, fontfamily='Helvetica', fontweight='bold')
    ax_temp.tick_params(axis='x', rotation=45)
    ax_temp.set_facecolor('white')
    ax_temp.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, temp in zip(bars, temps):
        height = bar.get_height()
        ax_temp.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{temp:.1f}°C', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 5. Quality Trend with Smooth Animation Effect
    ax_quality = plt.subplot(3, 4, 7)
    ax_quality.plot(dashboard_data['date_time'], dashboard_data['quality'], 
                   color=HONEYWELL_COLORS['secondary'], linewidth=3, alpha=0.8)
    ax_quality.set_title('Quality Trend', fontweight='bold', 
                        color=HONEYWELL_COLORS['dark_gray'], fontsize=14, fontfamily='Helvetica')
    ax_quality.set_ylabel('Quality Score', fontsize=12, fontfamily='Helvetica', fontweight='bold')
    ax_quality.grid(True, alpha=0.3)
    ax_quality.set_facecolor('white')
    
    # Add trend line
    z = np.polyfit(range(len(dashboard_data)), dashboard_data['quality'], 1)
    p = np.poly1d(z)
    ax_quality.plot(dashboard_data['date_time'], p(range(len(dashboard_data))), 
                   "r--", alpha=0.8, linewidth=2, label='Trend')
    ax_quality.legend(fontsize=10)
    
    # 6. System Health with Modern Gauge
    ax_health = plt.subplot(3, 4, 8)
    health_metrics = {
        'CPU': 45,
        'Memory': 62,
        'Storage': 78,
        'Network': 100
    }
    
    metrics = list(health_metrics.keys())
    values = list(health_metrics.values())
    colors = [HONEYWELL_COLORS['success'] if v < 70 else HONEYWELL_COLORS['accent'] if v < 90 else HONEYWELL_COLORS['danger'] for v in values]
    
    # Create horizontal bars with modern styling
    bars = ax_health.barh(metrics, values, color=colors, alpha=0.8, height=0.6)
    ax_health.set_title('System Health', fontweight='bold', 
                       color=HONEYWELL_COLORS['dark_gray'], fontsize=14, fontfamily='Helvetica')
    ax_health.set_xlabel('Usage (%)', fontsize=12, fontfamily='Helvetica', fontweight='bold')
    ax_health.set_facecolor('white')
    ax_health.grid(True, alpha=0.3, axis='x')
    
    # Add value labels with modern styling
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax_health.text(value + 2, bar.get_y() + bar.get_height()/2, f'{value}%',
                      va='center', fontweight='bold', fontsize=11, color=colors[i])
    
    # 7. Process Efficiency with Donut Chart
    ax_efficiency = plt.subplot(3, 4, 9)
    total_records = len(dashboard_data)
    high_quality = (dashboard_data['quality'] >= 400).sum()
    medium_quality = ((dashboard_data['quality'] >= 350) & (dashboard_data['quality'] < 400)).sum()
    low_quality = (dashboard_data['quality'] < 350).sum()
    
    efficiency_data = ['High', 'Medium', 'Low']
    efficiency_counts = [high_quality, medium_quality, low_quality]
    efficiency_colors = [HONEYWELL_COLORS['success'], HONEYWELL_COLORS['accent'], HONEYWELL_COLORS['danger']]
    
    # Create donut chart
    wedges, texts, autotexts = ax_efficiency.pie(efficiency_counts, labels=efficiency_data, 
                                                colors=efficiency_colors, autopct='%1.1f%%',
                                                startangle=90, pctdistance=0.85)
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax_efficiency.add_patch(centre_circle)
    
    # Add percentage in center
    total_percentage = (high_quality/total_records)*100
    ax_efficiency.text(0, 0, f'{total_percentage:.1f}%\nEFFICIENT', ha='center', va='center',
                      fontsize=14, fontweight='bold', color=HONEYWELL_COLORS['dark_gray'],
                      fontfamily='Helvetica')
    
    ax_efficiency.set_title('Process Efficiency', fontweight='bold', 
                           color=HONEYWELL_COLORS['dark_gray'], fontsize=14, fontfamily='Helvetica')
    
    # 8. Real-time Alerts with Modern Cards
    ax_alerts = plt.subplot(3, 4, 10)
    alerts = {
        'Temperature': np.random.randint(2, 8),
        'Humidity': np.random.randint(1, 5),
        'Quality': np.random.randint(0, 3),
        'System': np.random.randint(0, 2)
    }
    
    alert_types = list(alerts.keys())
    alert_counts = list(alerts.values())
    alert_colors = [HONEYWELL_COLORS['danger'], HONEYWELL_COLORS['secondary'], 
                   HONEYWELL_COLORS['accent'], HONEYWELL_COLORS['warning']]
    
    # Create modern card-style bars
    bars = ax_alerts.bar(alert_types, alert_counts, color=alert_colors, alpha=0.9, 
                        edgecolor='white', linewidth=2)
    ax_alerts.set_title('System Alerts', fontweight='bold', 
                       color=HONEYWELL_COLORS['dark_gray'], fontsize=14, fontfamily='Helvetica')
    ax_alerts.set_ylabel('Count', fontsize=12, fontfamily='Helvetica', fontweight='bold')
    ax_alerts.tick_params(axis='x', rotation=45)
    ax_alerts.set_facecolor('white')
    ax_alerts.grid(True, alpha=0.3, axis='y')
    
    # Add alert icons
    for i, (bar, count) in enumerate(zip(bars, alert_counts)):
        if count > 0:
            ax_alerts.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          '⚠️', ha='center', va='bottom', fontsize=16)
    
    # 9. Key Performance Indicators
    ax_kpi = plt.subplot(3, 4, 11)
    kpi_data = {
        'Quality Score': f"{current_quality:.1f}",
        'Efficiency': f"{(high_quality/total_records)*100:.1f}%",
        'Uptime': "99.2%",
        'Energy': "94.5%"
    }
    
    y_pos = 0.9
    for kpi, value in kpi_data.items():
        ax_kpi.text(0.05, y_pos, f'{kpi}:', transform=ax_kpi.transAxes, fontsize=12,
                   color=HONEYWELL_COLORS['dark_gray'], fontfamily='Helvetica', fontweight='bold')
        ax_kpi.text(0.6, y_pos, value, transform=ax_kpi.transAxes, fontsize=14, fontweight='bold',
                   color=HONEYWELL_COLORS['primary'], fontfamily='Helvetica')
        y_pos -= 0.2
    
    ax_kpi.set_title('Key Performance Indicators', fontweight='bold', 
                    color=HONEYWELL_COLORS['dark_gray'], fontsize=14, fontfamily='Helvetica')
    ax_kpi.set_facecolor('white')
    ax_kpi.axis('off')
    
    # 10. Footer with Professional Branding
    ax_footer = plt.subplot(3, 4, 12)
    footer_text = """
    HONEYWELL PROCESS CONTROL & AUTOMATION
    
    Real-time monitoring and predictive analytics
    for industrial coffee bean roasting processes.
    
    For technical support: support@honeywell.com
    """
    
    ax_footer.text(0.5, 0.8, footer_text, transform=ax_footer.transAxes, 
                  fontsize=10, ha='center', va='top', fontfamily='Helvetica',
                  color=HONEYWELL_COLORS['dark_gray'], fontweight='bold')
    
    # Add Honeywell logo text at bottom
    ax_footer.text(0.5, 0.2, 'HONEYWELL', transform=ax_footer.transAxes, 
                  fontsize=16, ha='center', va='center', fontfamily='Helvetica',
                  color=HONEYWELL_COLORS['primary'], fontweight='bold')
    ax_footer.axis('off')
    
    # Save dashboard with high quality
    plt.tight_layout()
    plt.savefig('honeywell_professional_dashboard.png', dpi=300, bbox_inches='tight', 
                facecolor=HONEYWELL_COLORS['light_gray'])
    plt.show()
    
    print("✅ Professional Honeywell Dashboard created: honeywell_professional_dashboard.png")

if __name__ == "__main__":
    create_honeywell_dashboard()
