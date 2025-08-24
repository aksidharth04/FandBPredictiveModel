import React from 'react'
import { 
  LayoutDashboard, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  Settings, 
  BarChart3,
  Thermometer,
  Gauge
} from 'lucide-react'

const Sidebar = () => {
  const menuItems = [
    { icon: LayoutDashboard, label: 'Dashboard', active: true },
    { icon: Activity, label: 'Real-time Monitoring' },
    { icon: TrendingUp, label: 'Quality Trends' },
    { icon: Thermometer, label: 'Temperature Zones' },
    { icon: Gauge, label: 'Process Efficiency' },
    { icon: AlertTriangle, label: 'Alerts & Warnings' },
    { icon: BarChart3, label: 'Analytics' },
    { icon: Settings, label: 'Settings' }
  ]

  return (
    <aside className="w-64 bg-white shadow-lg min-h-screen">
      <div className="p-6">
        <nav className="space-y-2">
          {menuItems.map((item, index) => (
            <a
              key={index}
              href="#"
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                item.active
                  ? 'bg-honeywell-primary text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
      </div>
    </aside>
  )
}

export default Sidebar
