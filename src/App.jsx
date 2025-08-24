import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import Dashboard from './components/Dashboard'

function App() {
  const [currentTime, setCurrentTime] = useState(new Date())
  const [isLive, setIsLive] = useState(true)

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <Header currentTime={currentTime} isLive={isLive} />
      
      {/* Main Content - Full Width */}
      <main className="p-6">
        <Dashboard currentTime={currentTime} isLive={isLive} />
      </main>
    </div>
  )
}

export default App
