import React from 'react'
import { AlertTriangle, Clock, XCircle, Info } from 'lucide-react'

const AlertsPanel = ({ alerts }) => {
  const alertTypes = [
    { type: 'Temperature', count: 2, severity: 'warning', icon: AlertTriangle },
    { type: 'Humidity', count: 1, severity: 'info', icon: Info },
    { type: 'Quality', count: 0, severity: 'success', icon: XCircle },
    { type: 'System', count: 0, severity: 'success', icon: XCircle }
  ]

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50'
      case 'warning': return 'text-orange-600 bg-orange-50'
      case 'info': return 'text-blue-600 bg-blue-50'
      case 'success': return 'text-green-600 bg-green-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return '🔴'
      case 'warning': return '🟡'
      case 'info': return '🔵'
      case 'success': return '🟢'
      default: return '⚪'
    }
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">System Alerts</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-600">Live</span>
        </div>
      </div>

      <div className="space-y-3">
        {alertTypes.map((alert, index) => (
          <div key={index} className="flex items-center justify-between p-3 rounded-lg border border-gray-200">
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${getSeverityColor(alert.severity).split(' ')[1]}`}>
                <alert.icon className={`w-4 h-4 ${getSeverityColor(alert.severity).split(' ')[0]}`} />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-900">{alert.type} Alerts</div>
                <div className="text-xs text-gray-500">
                  {alert.count > 0 ? `${alert.count} active` : 'No alerts'}
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-lg">{getSeverityIcon(alert.severity)}</span>
              <span className={`text-sm font-medium ${getSeverityColor(alert.severity).split(' ')[0]}`}>
                {alert.count}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Alerts */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Recent Alerts</h4>
        <div className="space-y-2">
          <div className="flex items-center space-x-2 text-sm">
            <Clock className="w-4 h-4 text-gray-400" />
            <span className="text-gray-600">2 min ago:</span>
            <span className="text-orange-600">Temperature deviation in Main Roasting zone</span>
          </div>
          <div className="flex items-center space-x-2 text-sm">
            <Clock className="w-4 h-4 text-gray-400" />
            <span className="text-gray-600">5 min ago:</span>
            <span className="text-blue-600">Humidity levels approaching threshold</span>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Total Active Alerts:</span>
          <span className="font-bold text-gray-900">{alerts}</span>
        </div>
      </div>
    </div>
  )
}

export default AlertsPanel
