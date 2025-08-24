import React from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'

const QualityTrend = () => {
  // Generate sample trend data
  const generateTrendData = () => {
    const data = []
    let currentValue = 420
    for (let i = 0; i < 24; i++) {
      currentValue += (Math.random() - 0.5) * 20
      data.push({
        time: `${i}:00`,
        value: Math.max(350, Math.min(500, currentValue))
      })
    }
    return data
  }

  const trendData = generateTrendData()
  const currentValue = trendData[trendData.length - 1].value
  const previousValue = trendData[trendData.length - 2].value
  const trend = currentValue > previousValue ? 'up' : 'down'
  const trendPercent = Math.abs(((currentValue - previousValue) / previousValue) * 100)

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">Quality Trend</h3>
        <div className={`flex items-center space-x-1 text-sm ${
          trend === 'up' ? 'text-green-600' : 'text-red-600'
        }`}>
          {trend === 'up' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span>{trendPercent.toFixed(1)}%</span>
        </div>
      </div>

      {/* Simple trend visualization */}
      <div className="space-y-2">
        {trendData.slice(-8).map((point, index) => (
          <div key={index} className="flex items-center justify-between">
            <div className="text-xs text-gray-500 w-12">{point.time}</div>
            <div className="flex-1 mx-2">
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full"
                  style={{ width: `${((point.value - 350) / 150) * 100}%` }}
                ></div>
              </div>
            </div>
            <div className="text-xs font-medium text-gray-900 w-12 text-right">
              {point.value.toFixed(0)}
            </div>
          </div>
        ))}
      </div>

      {/* Current status */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Current Quality:</span>
          <span className="text-lg font-bold text-gray-900">{currentValue.toFixed(1)}</span>
        </div>
        <div className="flex items-center justify-between mt-1">
          <span className="text-sm text-gray-600">Status:</span>
          <span className={`text-sm font-medium ${
            currentValue >= 450 ? 'text-green-600' : 
            currentValue >= 400 ? 'text-blue-600' : 
            currentValue >= 350 ? 'text-yellow-600' : 'text-red-600'
          }`}>
            {currentValue >= 450 ? 'Excellent' : 
             currentValue >= 400 ? 'Good' : 
             currentValue >= 350 ? 'Acceptable' : 'Poor'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default QualityTrend
