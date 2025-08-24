import React from 'react'
import { TrendingUp, CheckCircle } from 'lucide-react'

const KPICards = ({ data }) => {
  const kpiData = [
    {
      title: 'Quality Score',
      value: data.quality.toFixed(1),
      unit: '',
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200'
    },
    {
      title: 'Process Efficiency',
      value: data.efficiency.toFixed(1),
      unit: '%',
      icon: TrendingUp,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {kpiData.map((kpi, index) => (
        <div key={index} className={`metric-card ${kpi.borderColor} ${kpi.bgColor}`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">{kpi.title}</p>
              <p className="text-2xl font-bold text-gray-900">
                {kpi.value}{kpi.unit}
              </p>
            </div>
            <div className={`p-3 rounded-full ${kpi.bgColor}`}>
              <kpi.icon className={`w-6 h-6 ${kpi.color}`} />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default KPICards
