import React from 'react'
import { Activity, Clock, Wifi } from 'lucide-react'

const Header = ({ currentTime, isLive }) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-honeywell-primary rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">H</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-honeywell-primary">HONEYWELL</h1>
                <p className="text-sm text-gray-600 font-medium">Process Control & Automation</p>
              </div>
            </div>
          </div>

          {/* Center Title */}
          <div className="text-center">
            <h2 className="text-lg font-bold text-honeywell-dark-gray">
              Coffee Roasting Process Monitoring
            </h2>
            <p className="text-sm text-gray-500">Real-time Anomaly Detection & Quality Prediction</p>
          </div>

          {/* Status and Time */}
          <div className="flex items-center space-x-6">
            {/* Live Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
              <span className="text-sm font-medium text-gray-700">
                {isLive ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>

            {/* Current Time */}
            <div className="flex items-center space-x-2 text-gray-600">
              <Clock className="w-4 h-4" />
              <span className="text-sm font-medium">
                {currentTime.toLocaleTimeString()}
              </span>
            </div>

            {/* Connection Status */}
            <div className="flex items-center space-x-2 text-green-600">
              <Wifi className="w-4 h-4" />
              <span className="text-sm font-medium">Connected</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
