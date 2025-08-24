import React from 'react'
import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react'

const QualityGauge = ({ quality }) => {
  // Calculate quality percentage (assuming range 221-505)
  const minQuality = 221
  const maxQuality = 505
  const qualityPercent = ((quality - minQuality) / (maxQuality - minQuality)) * 100

  // Determine quality status
  let status = 'excellent'
  let statusColor = 'text-green-600'
  let statusBg = 'bg-green-50'
  let statusIcon = CheckCircle
  let statusText = 'EXCELLENT'

  if (qualityPercent < 40) {
    status = 'poor'
    statusColor = 'text-red-600'
    statusBg = 'bg-red-50'
    statusIcon = XCircle
    statusText = 'POOR'
  } else if (qualityPercent < 60) {
    status = 'acceptable'
    statusColor = 'text-yellow-600'
    statusBg = 'bg-yellow-50'
    statusIcon = AlertTriangle
    statusText = 'ACCEPTABLE'
  } else if (qualityPercent < 80) {
    status = 'good'
    statusColor = 'text-blue-600'
    statusBg = 'bg-blue-50'
    statusIcon = CheckCircle
    statusText = 'GOOD'
  }

  // Calculate circle circumference and stroke dasharray
  const radius = 60
  const circumference = 2 * Math.PI * radius
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (qualityPercent / 100) * circumference

  return (
    <div className="card">
      <div className="text-center">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Current Quality</h3>
        
        {/* Circular Progress */}
        <div className="relative inline-block">
          <svg className="w-32 h-32 transform -rotate-90">
            {/* Background circle */}
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke="#e5e7eb"
              strokeWidth="8"
              fill="transparent"
            />
            {/* Progress circle */}
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke={status === 'excellent' ? '#10b981' : status === 'good' ? '#3b82f6' : status === 'acceptable' ? '#f59e0b' : '#ef4444'}
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
            />
          </svg>
          
          {/* Center content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-2xl font-bold text-gray-900">{quality.toFixed(1)}</div>
            <div className="text-sm text-gray-500">{qualityPercent.toFixed(1)}%</div>
          </div>
        </div>

        {/* Status indicator */}
        <div className={`mt-4 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusBg} ${statusColor}`}>
          <statusIcon className="w-4 h-4 mr-2" />
          {statusText}
        </div>

        {/* Quality range info */}
        <div className="mt-4 text-xs text-gray-500">
          <p>Range: {minQuality} - {maxQuality}</p>
          <p>Target: 400+ (Excellent)</p>
        </div>
      </div>
    </div>
  )
}

export default QualityGauge
