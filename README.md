# 🚀 Honeywell F&B Process Monitoring Dashboard

**Professional real-time monitoring dashboard for industrial coffee bean roasting processes**  
_Built with React, Tailwind CSS, and modern web technologies_

## ✨ Features

* 🎯 **Real-time Monitoring**: Live quality score tracking and process efficiency metrics
* 📊 **Interactive Dashboards**: Professional widgets with animated charts and gauges
* 🔔 **Smart Alerts**: Real-time alert system with severity-based notifications
* 🧠 **AI Integration**: Model training performance visualization
* 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
* 🎨 **Modern UI**: Clean, professional interface with Honeywell branding

## 🛠️ Tech Stack

**Frontend**: React 18, Vite, Tailwind CSS  
**Charts**: Custom SVG-based visualizations  
**Icons**: Lucide React  
**Styling**: Tailwind CSS with custom Honeywell color palette

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd honeywell-fb-dashboard

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:5173` to access the dashboard!

### Build for Production

```bash
# Build the application
npm run build

# Preview the build
npm run preview
```

## 📁 Project Structure

```
src/
├── components/
│   ├── Header.jsx              # Main header with Honeywell branding
│   ├── Sidebar.jsx             # Navigation sidebar
│   ├── Dashboard.jsx           # Main dashboard layout
│   └── widgets/                # Individual dashboard widgets
│       ├── KPICards.jsx        # Key performance indicator cards
│       ├── QualityGauge.jsx    # Circular quality gauge
│       ├── ModelTraining.jsx   # AI model training visualization
│       ├── TemperatureChart.jsx # Temperature zone monitoring
│       ├── QualityTrend.jsx    # Quality trend analysis
│       ├── SystemHealth.jsx    # System health metrics
│       ├── ProcessEfficiency.jsx # Process efficiency donut chart
│       └── AlertsPanel.jsx     # Real-time alerts panel
├── App.jsx                     # Main application component
├── main.jsx                    # Application entry point
└── index.css                   # Global styles and Tailwind imports
```

## 🎨 Design Features

### Honeywell Branding
- **Primary Color**: Honeywell Red (#D32F2F)
- **Secondary Color**: Blue (#1976D2)
- **Typography**: Helvetica font family
- **Professional Layout**: Clean, modern interface

### Interactive Widgets
- **Quality Gauge**: Animated circular progress indicator
- **Temperature Monitoring**: Real-time zone temperature tracking
- **System Health**: Live system metrics with status indicators
- **Alert System**: Severity-based alert management
- **Process Efficiency**: Donut chart with breakdown metrics

### Real-time Features
- **Live Updates**: Simulated real-time data updates every 5 seconds
- **Status Indicators**: Animated status dots and progress bars
- **Responsive Grid**: Adaptive layout for different screen sizes

## 🔧 Configuration

### Customizing Colors
Edit `tailwind.config.js` to modify the Honeywell color palette:

```javascript
colors: {
  honeywell: {
    primary: '#D32F2F',      // Honeywell Red
    secondary: '#1976D2',    // Blue
    accent: '#FFC107',       // Amber
    success: '#388E3C',      // Green
    warning: '#F57C00',      // Orange
    danger: '#D32F2F',       // Red
    'light-gray': '#F5F5F5', // Light Gray
    'dark-gray': '#424242'   // Dark Gray
  }
}
```

### Adding New Widgets
1. Create a new component in `src/components/widgets/`
2. Import and add it to the Dashboard grid in `Dashboard.jsx`
3. Follow the existing widget patterns for consistency

## 📊 Dashboard Components

### KPI Cards
- Quality Score with real-time updates
- Process Efficiency percentage
- System Uptime monitoring
- Active Alerts count

### Quality Monitoring
- Circular gauge with percentage display
- Quality status indicators (Excellent/Good/Acceptable/Poor)
- Real-time quality trend analysis

### Temperature Zones
- Individual zone temperature monitoring
- Target range indicators
- Average temperature calculation

### System Health
- CPU, Memory, Storage, and Network usage
- Health status indicators
- Overall system health summary

### Process Efficiency
- Donut chart visualization
- Quality breakdown (High/Medium/Low)
- Efficiency status indicators

### Alert Management
- Real-time alert monitoring
- Severity-based alert categorization
- Recent alert history

## 🎯 Key Features

### Real-time Data Simulation
The dashboard includes simulated real-time data updates to demonstrate live monitoring capabilities:

```javascript
useEffect(() => {
  const interval = setInterval(() => {
    setDashboardData(prev => ({
      ...prev,
      quality: prev.quality + (Math.random() - 0.5) * 10,
      // ... other metrics
    }))
  }, 5000)
  return () => clearInterval(interval)
}, [])
```

### Responsive Design
- Mobile-first approach
- Adaptive grid layouts
- Touch-friendly interface
- Optimized for all screen sizes

### Professional Styling
- Consistent Honeywell branding
- Modern card-based design
- Smooth animations and transitions
- Professional color scheme

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **MIT License**.

## 👨‍💻 Author

**Adicherikandi Sidharth**

* GitHub: [@aksidharth04](https://github.com/aksidharth04)
* LinkedIn: [Adicherikandi Sidharth](https://linkedin.com/in/aksidharth)
* Email: aksidharthm10@gmail.com

## 🆘 Support

* Email: aksidharthm10@gmail.com
* GitHub Issues: Create an issue for bugs or feature requests

---

**Made with ❤️ by Adicherikandi Sidharth**

_Honeywell F&B Process Monitoring Dashboard - Professional industrial monitoring solution_ 🚀
