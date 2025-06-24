
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import { Brain, Target, Zap, TrendingUp, BarChart3, Activity } from 'lucide-react';

interface FactorModel {
  factor: string;
  beta: number;
  tStat: number;
  pValue: number;
  rSquared: number;
}

interface BacktestResults {
  strategy: string;
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  trades: number;
}

export const QuantResearch = () => {
  const [activeModel, setActiveModel] = useState('fama-french');
  const [factorModels, setFactorModels] = useState<FactorModel[]>([
    { factor: 'Market (Rm-Rf)', beta: 0.95, tStat: 12.5, pValue: 0.001, rSquared: 0.72 },
    { factor: 'Size (SMB)', beta: -0.15, tStat: -2.8, pValue: 0.006, rSquared: 0.08 },
    { factor: 'Value (HML)', beta: 0.22, tStat: 3.2, pValue: 0.002, rSquared: 0.12 },
    { factor: 'Profitability (RMW)', beta: 0.18, tStat: 2.1, pValue: 0.038, rSquared: 0.05 },
    { factor: 'Investment (CMA)', beta: -0.12, tStat: -1.9, pValue: 0.062, rSquared: 0.03 }
  ]);

  const [backtestResults, setBacktestResults] = useState<BacktestResults[]>([
    {
      strategy: 'Mean Reversion',
      totalReturn: 0.185,
      annualizedReturn: 0.142,
      volatility: 0.165,
      sharpeRatio: 0.86,
      maxDrawdown: -0.085,
      winRate: 0.58,
      trades: 245
    },
    {
      strategy: 'Momentum',
      totalReturn: 0.235,
      annualizedReturn: 0.178,
      volatility: 0.195,
      sharpeRatio: 0.91,
      maxDrawdown: -0.125,
      winRate: 0.52,
      trades: 189
    },
    {
      strategy: 'Statistical Arbitrage',
      totalReturn: 0.165,
      annualizedReturn: 0.128,
      volatility: 0.098,
      sharpeRatio: 1.31,
      maxDrawdown: -0.045,
      winRate: 0.68,
      trades: 1250
    }
  ]);

  const [performanceData, setPerformanceData] = useState([
    { date: '2024-01', strategy: 0.02, benchmark: 0.015, alpha: 0.005 },
    { date: '2024-02', strategy: 0.035, benchmark: 0.025, alpha: 0.01 },
    { date: '2024-03', strategy: -0.015, benchmark: -0.025, alpha: 0.01 },
    { date: '2024-04', strategy: 0.045, benchmark: 0.038, alpha: 0.007 },
    { date: '2024-05', strategy: 0.028, benchmark: 0.022, alpha: 0.006 },
    { date: '2024-06', strategy: 0.052, benchmark: 0.041, alpha: 0.011 }
  ]);

  const models = [
    { id: 'fama-french', name: 'Fama-French 5-Factor' },
    { id: 'capm', name: 'CAPM' },
    { id: 'apt', name: 'Arbitrage Pricing Theory' },
    { id: 'black-litterman', name: 'Black-Litterman' }
  ];

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">QUANTITATIVE RESEARCH</h2>
          </div>
          <div className="flex space-x-2">
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => setActiveModel(model.id)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  activeModel === model.id 
                    ? 'bg-terminal-orange text-terminal-bg' 
                    : 'bg-terminal-border text-terminal-muted hover:bg-terminal-orange/20'
                }`}
              >
                {model.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Factor Model Results */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">
            FACTOR MODEL: {models.find(m => m.id === activeModel)?.name}
          </h3>
          <div className="space-y-2">
            {factorModels.map((factor, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded">
                <div className="flex-1">
                  <div className="text-xs font-medium text-terminal-text">{factor.factor}</div>
                  <div className="text-xs text-terminal-muted">
                    R² = {factor.rSquared.toFixed(3)}
                  </div>
                </div>
                <div className="flex space-x-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Beta</div>
                    <div className={`font-semibold financial-number ${
                      factor.beta > 0 ? 'text-terminal-green' : 'text-terminal-red'
                    }`}>
                      {factor.beta.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">t-Stat</div>
                    <div className={`font-semibold financial-number ${
                      Math.abs(factor.tStat) > 2 ? 'text-terminal-green' : 'text-terminal-amber'
                    }`}>
                      {factor.tStat.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">p-Value</div>
                    <div className={`font-semibold financial-number ${
                      factor.pValue < 0.05 ? 'text-terminal-green' : 'text-terminal-red'
                    }`}>
                      {factor.pValue.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Strategy Performance Chart */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">STRATEGY vs BENCHMARK</h3>
          <div className="h-32">
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
                  tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                />
                <Line 
                  type="monotone" 
                  dataKey="strategy" 
                  stroke="#00ff88" 
                  strokeWidth={2}
                  dot={false}
                  name="Strategy"
                />
                <Line 
                  type="monotone" 
                  dataKey="benchmark" 
                  stroke="#888888" 
                  strokeWidth={2}
                  dot={false}
                  name="Benchmark"
                />
                <Line 
                  type="monotone" 
                  dataKey="alpha" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                  name="Alpha"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Backtest Results */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">STRATEGY BACKTESTS</h3>
          <div className="space-y-3">
            {backtestResults.map((result, index) => (
              <div key={index} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-terminal-text">{result.strategy}</h4>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-terminal-muted">{result.trades} trades</span>
                    <div className={`w-2 h-2 rounded-full ${
                      result.sharpeRatio > 1 ? 'bg-terminal-green' : 
                      result.sharpeRatio > 0.5 ? 'bg-terminal-amber' : 'bg-terminal-red'
                    }`} />
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Total Return</div>
                    <div className="text-terminal-green font-semibold financial-number">
                      {(result.totalReturn * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Sharpe Ratio</div>
                    <div className="text-terminal-cyan font-semibold financial-number">
                      {result.sharpeRatio.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Max DD</div>
                    <div className="text-terminal-red font-semibold financial-number">
                      {(result.maxDrawdown * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Volatility</div>
                    <div className="text-terminal-text font-semibold financial-number">
                      {(result.volatility * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Win Rate</div>
                    <div className="text-terminal-green font-semibold financial-number">
                      {(result.winRate * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Ann. Return</div>
                    <div className="text-terminal-green font-semibold financial-number">
                      {(result.annualizedReturn * 100).toFixed(1)}%
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
