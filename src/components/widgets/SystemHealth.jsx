import React from 'react'
import { Activity, Cpu, HardDrive, Wifi } from 'lucide-react'

const SystemHealth = () => {
  const healthMetrics = [
    { name: 'CPU Usage', value: 45, icon: Cpu, color: 'text-blue-600', bgColor: 'bg-blue-50' },
    { name: 'Memory Usage', value: 62, icon: Activity, color: 'text-green-600', bgColor: 'bg-green-50' },
    { name: 'Storage Usage', value: 78, icon: HardDrive, color: 'text-orange-600', bgColor: 'bg-orange-50' },
    { name: 'Network Status', value: 100, icon: Wifi, color: 'text-purple-600', bgColor: 'bg-purple-50' }
  ]

  const getStatusColor = (value) => {
    if (value < 70) return 'text-green-600'
    if (value < 90) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getStatusBg = (value) => {
    if (value < 70) return 'bg-green-50'
    if (value < 90) return 'bg-yellow-50'
    return 'bg-red-50'
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">System Health</h3>
        <div className="flex items-center space-x-1 text-green-600">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-medium">Healthy</span>
        </div>
      </div>

      <div className="space-y-4">
        {healthMetrics.map((metric, index) => (
          <div key={index} className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                <metric.icon className={`w-4 h-4 ${metric.color}`} />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">{metric.name}</div>
                <div className="text-xs text-gray-500">
                  {metric.value < 70 ? 'Optimal' : metric.value < 90 ? 'Warning' : 'Critical'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className={`text-lg font-bold ${getStatusColor(metric.value)}`}>
                {metric.value}%
              </div>
              <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full ${
                    metric.value < 70 ? 'bg-green-500' : 
                    metric.value < 90 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${metric.value}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* System Status Summary */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Overall Health:</span>
            <span className="ml-2 font-bold text-green-600">Excellent</span>
          </div>
          <div>
            <span className="text-gray-600">Uptime:</span>
            <span className="ml-2 font-bold text-gray-900">99.2%</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SystemHealth
