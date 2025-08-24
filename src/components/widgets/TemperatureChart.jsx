import React from 'react'
import { Thermometer, TrendingUp } from 'lucide-react'

const TemperatureChart = ({ data }) => {
  const zones = [
    { name: 'Drying', temp: data.drying, color: 'bg-blue-500', target: '150-220°C' },
    { name: 'Pre-Roasting', temp: data.preRoasting, color: 'bg-green-500', target: '220-380°C' },
    { name: 'Main-Roasting', temp: data.mainRoasting, color: 'bg-orange-500', target: '380-520°C' },
    { name: 'Post-Roasting', temp: data.postRoasting, color: 'bg-purple-500', target: '300-450°C' },
    { name: 'Cooling', temp: data.cooling, color: 'bg-red-500', target: '200-300°C' }
  ]

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">Zone Temperatures</h3>
        <Thermometer className="w-5 h-5 text-gray-500" />
      </div>

      <div className="space-y-4">
        {zones.map((zone, index) => (
          <div key={index} className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${zone.color}`}></div>
              <div>
                <div className="text-sm font-medium text-gray-900">{zone.name}</div>
                <div className="text-xs text-gray-500">Target: {zone.target}</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-gray-900">{zone.temp.toFixed(1)}°C</div>
              <div className="text-xs text-gray-500">
                {zone.temp > 400 ? 'High' : zone.temp > 300 ? 'Medium' : 'Low'}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Average Temperature:</span>
          <span className="font-bold text-gray-900">
            {(Object.values(data).reduce((a, b) => a + b, 0) / Object.values(data).length).toFixed(1)}°C
          </span>
        </div>
      </div>
    </div>
  )
}

export default TemperatureChart
