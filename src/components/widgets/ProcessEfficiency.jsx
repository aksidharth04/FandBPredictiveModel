import React from 'react'
import { TrendingUp, Target } from 'lucide-react'

const ProcessEfficiency = ({ efficiency }) => {
  // Calculate donut chart values
  const radius = 50
  const circumference = 2 * Math.PI * radius
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (efficiency / 100) * circumference

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">Process Efficiency</h3>
        <Target className="w-5 h-5 text-gray-500" />
      </div>

      <div className="text-center">
        {/* Donut Chart */}
        <div className="relative inline-block mb-4">
          <svg className="w-32 h-32 transform -rotate-90">
            {/* Background circle */}
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke="#e5e7eb"
              strokeWidth="12"
              fill="transparent"
            />
            {/* Progress circle */}
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke={efficiency >= 90 ? '#10b981' : efficiency >= 70 ? '#3b82f6' : '#f59e0b'}
              strokeWidth="12"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
            />
          </svg>
          
          {/* Center content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-2xl font-bold text-gray-900">{efficiency.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">Efficient</div>
          </div>
        </div>

        {/* Efficiency breakdown */}
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="text-center">
            <div className="text-lg font-bold text-green-600">92%</div>
            <div className="text-xs text-gray-500">High Quality</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">6%</div>
            <div className="text-xs text-gray-500">Medium</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-orange-600">2%</div>
            <div className="text-xs text-gray-500">Low</div>
          </div>
        </div>

        {/* Status indicator */}
        <div className="mt-4 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-50 text-green-800">
          <TrendingUp className="w-4 h-4 mr-2" />
          {efficiency >= 90 ? 'Excellent' : efficiency >= 70 ? 'Good' : 'Needs Attention'}
        </div>
      </div>
    </div>
  )
}

export default ProcessEfficiency
