import React, { useState, useEffect } from 'react'
import QualityGauge from './widgets/QualityGauge'
import TemperatureChart from './widgets/TemperatureChart'
import ProcessEfficiency from './widgets/ProcessEfficiency'
import KPICards from './widgets/KPICards'
import ModelTraining from './widgets/ModelTraining'

const Dashboard = ({ currentTime, isLive }) => {
  const [dashboardData, setDashboardData] = useState({
    quality: 425.6,
    temperature: {
      drying: 185,
      preRoasting: 320,
      mainRoasting: 450,
      postRoasting: 380,
      cooling: 250
    },
    efficiency: 87.5
  })

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setDashboardData(prev => ({
        ...prev,
        quality: prev.quality + (Math.random() - 0.5) * 10,
        temperature: {
          drying: prev.temperature.drying + (Math.random() - 0.5) * 5,
          preRoasting: prev.temperature.preRoasting + (Math.random() - 0.5) * 8,
          mainRoasting: prev.temperature.mainRoasting + (Math.random() - 0.5) * 10,
          postRoasting: prev.temperature.postRoasting + (Math.random() - 0.5) * 6,
          cooling: prev.temperature.cooling + (Math.random() - 0.5) * 4
        }
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Process Monitoring Dashboard</h1>
          <p className="text-gray-600">Real-time monitoring of coffee roasting process</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm text-gray-500">Last Updated</p>
            <p className="text-sm font-medium">{currentTime.toLocaleTimeString()}</p>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <KPICards data={dashboardData} />

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Quality Gauge */}
        <div className="lg:col-span-1">
          <QualityGauge quality={dashboardData.quality} />
        </div>

        {/* Model Training Performance - Large */}
        <div className="lg:col-span-2">
          <ModelTraining />
        </div>

        {/* Temperature Chart */}
        <div className="lg:col-span-1">
          <TemperatureChart data={dashboardData.temperature} />
        </div>

        {/* Process Efficiency */}
        <div className="lg:col-span-2">
          <ProcessEfficiency efficiency={dashboardData.efficiency} />
        </div>
      </div>
    </div>
  )
}

export default Dashboard
