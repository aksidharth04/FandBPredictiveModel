import React from 'react'
import { Brain, TrendingUp, CheckCircle } from 'lucide-react'

const ModelTraining = () => {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">Model Training Performance</h3>
        <div className="flex items-center space-x-2 text-green-600">
          <CheckCircle className="w-4 h-4" />
          <span className="text-sm font-medium">Optimized</span>
        </div>
      </div>

      {/* Training Performance Image */}
      <div className="bg-gray-50 rounded-lg p-4 border-2 border-dashed border-gray-200">
        <div className="text-center">
          <img 
            src="/coffee_roasting_optimized_training.png" 
            alt="Coffee Roasting Model Training Performance"
            className="w-full h-auto rounded-lg shadow-sm"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'block';
            }}
          />
          <div className="hidden text-center">
            <Brain className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <h4 className="text-sm font-medium text-gray-600 mb-2">Coffee Roasting Model</h4>
            <p className="text-xs text-gray-500 mb-4">
              LSTM-based anomaly prediction model with optimized training curves
            </p>
            
            {/* Fallback content */}
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-center justify-center space-x-2 text-gray-500">
                <TrendingUp className="w-5 h-5" />
                <span className="text-sm">Training Performance Visualization</span>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Model accuracy: 94.2% | Loss: 0.023 | Validation: 92.8%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Model Stats */}
      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">94.2%</div>
          <div className="text-xs text-gray-500">Accuracy</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">0.023</div>
          <div className="text-xs text-gray-500">Loss</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-purple-600">92.8%</div>
          <div className="text-xs text-gray-500">Validation</div>
        </div>
      </div>
    </div>
  )
}

export default ModelTraining
