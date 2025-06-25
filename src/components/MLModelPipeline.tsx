
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { Brain, Zap, Target, TrendingUp, Activity, Cpu, Database } from 'lucide-react';

interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  lastTrained: string;
  status: 'training' | 'deployed' | 'testing' | 'error';
  predictions: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  correlation: number;
  stability: number;
}

interface PredictionResult {
  symbol: string;
  prediction: number;
  confidence: number;
  signal: 'buy' | 'sell' | 'hold';
  modelConsensus: number;
}

export const MLModelPipeline = () => {
  const [models, setModels] = useState<ModelMetrics[]>([
    { name: 'LSTM Price Predictor', accuracy: 0.847, precision: 0.823, recall: 0.865, f1Score: 0.844, lastTrained: '2024-06-25 08:30', status: 'deployed', predictions: 1542 },
    { name: 'XGBoost Momentum', accuracy: 0.792, precision: 0.788, recall: 0.796, f1Score: 0.792, lastTrained: '2024-06-25 06:15', status: 'deployed', predictions: 892 },
    { name: 'Random Forest Vol', accuracy: 0.731, precision: 0.745, recall: 0.718, f1Score: 0.731, lastTrained: '2024-06-25 09:45', status: 'training', predictions: 0 },
    { name: 'Transformer Sentiment', accuracy: 0.889, precision: 0.901, recall: 0.877, f1Score: 0.889, lastTrained: '2024-06-25 07:20', status: 'deployed', predictions: 2156 },
    { name: 'RL Strategy Optimizer', accuracy: 0.712, precision: 0.698, recall: 0.726, f1Score: 0.712, lastTrained: '2024-06-25 10:00', status: 'testing', predictions: 345 }
  ]);

  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([
    { feature: 'RSI_14', importance: 0.185, correlation: -0.42, stability: 0.91 },
    { feature: 'Volume_MA_20', importance: 0.156, correlation: 0.35, stability: 0.88 },
    { feature: 'Price_MA_Cross', importance: 0.142, correlation: 0.58, stability: 0.94 },
    { feature: 'MACD_Signal', importance: 0.128, correlation: 0.41, stability: 0.82 },
    { feature: 'Bollinger_Position', importance: 0.115, correlation: -0.28, stability: 0.76 },
    { feature: 'News_Sentiment', importance: 0.098, correlation: 0.22, stability: 0.65 },
    { feature: 'Options_Flow', importance: 0.087, correlation: 0.31, stability: 0.72 },
    { feature: 'Sector_Momentum', importance: 0.089, correlation: 0.19, stability: 0.84 }
  ]);

  const [predictions, setPredictions] = useState<PredictionResult[]>([
    { symbol: 'AAPL', prediction: 0.73, confidence: 0.89, signal: 'buy', modelConsensus: 0.85 },
    { symbol: 'GOOGL', prediction: 0.45, confidence: 0.67, signal: 'hold', modelConsensus: 0.52 },
    { symbol: 'TSLA', prediction: -0.62, confidence: 0.78, signal: 'sell', modelConsensus: 0.71 },
    { symbol: 'MSFT', prediction: 0.58, confidence: 0.82, signal: 'buy', modelConsensus: 0.79 },
    { symbol: 'NVDA', prediction: 0.91, confidence: 0.94, signal: 'buy', modelConsensus: 0.92 }
  ]);

  const [performanceData, setPerformanceData] = useState([
    { date: '2024-01', accuracy: 0.82, sharpe: 1.45, returns: 0.08 },
    { date: '2024-02', accuracy: 0.79, sharpe: 1.38, returns: 0.06 },
    { date: '2024-03', accuracy: 0.85, sharpe: 1.52, returns: 0.12 },
    { date: '2024-04', accuracy: 0.81, sharpe: 1.41, returns: 0.09 },
    { date: '2024-05', accuracy: 0.87, sharpe: 1.58, returns: 0.14 },
    { date: '2024-06', accuracy: 0.84, sharpe: 1.49, returns: 0.11 }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setModels(prev => prev.map(model => ({
        ...model,
        accuracy: model.accuracy + (Math.random() - 0.5) * 0.01,
        predictions: model.status === 'deployed' ? model.predictions + Math.floor(Math.random() * 10) : model.predictions
      })));

      setPredictions(prev => prev.map(pred => ({
        ...pred,
        prediction: pred.prediction + (Math.random() - 0.5) * 0.1,
        confidence: Math.min(0.99, Math.max(0.1, pred.confidence + (Math.random() - 0.5) * 0.05))
      })));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'text-terminal-green';
      case 'training': return 'text-terminal-amber';
      case 'testing': return 'text-terminal-cyan';
      case 'error': return 'text-terminal-red';
      default: return 'text-terminal-muted';
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'buy': return 'text-terminal-green';
      case 'sell': return 'text-terminal-red';
      case 'hold': return 'text-terminal-amber';
      default: return 'text-terminal-muted';
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">ML MODEL PIPELINE</h2>
          </div>
          <div className="flex items-center space-x-2">
            <Cpu className="w-3 h-3 text-terminal-green" />
            <span className="text-xs text-terminal-green">GPU ACTIVE</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Model Status */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">MODEL STATUS</h3>
          <div className="space-y-2">
            {models.map((model, index) => (
              <div key={index} className="bg-terminal-bg/30 p-2 rounded">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      model.status === 'deployed' ? 'bg-terminal-green' :
                      model.status === 'training' ? 'bg-terminal-amber animate-pulse' :
                      model.status === 'testing' ? 'bg-terminal-cyan' : 'bg-terminal-red'
                    }`} />
                    <span className="text-xs font-medium text-terminal-text">{model.name}</span>
                  </div>
                  <div className={`text-xs font-medium ${getStatusColor(model.status)}`}>
                    {model.status.toUpperCase()}
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <div className="text-terminal-muted">Accuracy</div>
                    <div className="text-terminal-green font-semibold financial-number">
                      {(model.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">F1 Score</div>
                    <div className="text-terminal-cyan font-semibold financial-number">
                      {model.f1Score.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Predictions</div>
                    <div className="text-terminal-text font-semibold financial-number">
                      {model.predictions.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Last Trained</div>
                    <div className="text-terminal-muted">
                      {model.lastTrained.split(' ')[1]}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Feature Importance */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">FEATURE IMPORTANCE</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportance}>
                <XAxis 
                  dataKey="feature" 
                  axisLine={false}
                  tick={{ fontSize: 9, fill: '#888888' }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <Bar dataKey="importance" fill="#00d4ff" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Performance */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">MODEL PERFORMANCE</h3>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <XAxis 
                  dataKey="date" 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#00ff88" 
                  strokeWidth={2}
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="sharpe" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Live Predictions */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">LIVE PREDICTIONS</h3>
          <div className="space-y-2">
            {predictions.map((pred, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-medium text-terminal-text">{pred.symbol}</span>
                  <div className={`text-xs font-bold ${getSignalColor(pred.signal)}`}>
                    {pred.signal.toUpperCase()}
                  </div>
                </div>
                <div className="flex space-x-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Prediction</div>
                    <div className={`font-semibold financial-number ${
                      pred.prediction > 0 ? 'text-terminal-green' : 'text-terminal-red'
                    }`}>
                      {pred.prediction > 0 ? '+' : ''}{(pred.prediction * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Confidence</div>
                    <div className="text-terminal-cyan font-semibold financial-number">
                      {(pred.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Consensus</div>
                    <div className="text-terminal-amber font-semibold financial-number">
                      {(pred.modelConsensus * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
