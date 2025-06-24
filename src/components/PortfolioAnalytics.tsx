
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, TrendingDown, Target, Shield, Zap, BarChart3, PieChart, Activity } from 'lucide-react';

interface PortfolioMetrics {
  totalValue: number;
  sharpeRatio: number;
  beta: number;
  alpha: number;
  var95: number;
  cvar: number;
  maxDrawdown: number;
  calmarRatio: number;
  sortinoRatio: number;
  informationRatio: number;
  trackingError: number;
  treynorRatio: number;
}

interface PerformanceAttribution {
  factor: string;
  contribution: number;
  weight: number;
  return: number;
}

interface RiskDecomposition {
  factor: string;
  risk: number;
  contribution: number;
  percentage: number;
}

export const PortfolioAnalytics = () => {
  const [metrics, setMetrics] = useState<PortfolioMetrics>({
    totalValue: 12500000,
    sharpeRatio: 1.85,
    beta: 0.95,
    alpha: 0.035,
    var95: -185000,
    cvar: -245000,
    maxDrawdown: -0.082,
    calmarRatio: 2.15,
    sortinoRatio: 2.42,
    informationRatio: 0.85,
    trackingError: 0.045,
    treynorRatio: 0.125
  });

  const [attribution, setAttribution] = useState<PerformanceAttribution[]>([
    { factor: 'Market Beta', contribution: 0.068, weight: 0.85, return: 0.08 },
    { factor: 'Value Factor', contribution: 0.025, weight: 0.15, return: 0.167 },
    { factor: 'Quality Factor', contribution: 0.018, weight: 0.12, return: 0.15 },
    { factor: 'Low Vol Factor', contribution: -0.008, weight: 0.08, return: -0.10 },
    { factor: 'Momentum Factor', contribution: 0.032, weight: 0.18, return: 0.178 },
    { factor: 'Size Factor', contribution: -0.012, weight: 0.05, return: -0.24 }
  ]);

  const [riskDecomp, setRiskDecomp] = useState<RiskDecomposition[]>([
    { factor: 'Market Risk', risk: 0.145, contribution: 0.68, percentage: 68 },
    { factor: 'Sector Risk', risk: 0.032, contribution: 0.15, percentage: 15 },
    { factor: 'Style Risk', risk: 0.021, contribution: 0.10, percentage: 10 },
    { factor: 'Specific Risk', risk: 0.015, contribution: 0.07, percentage: 7 }
  ]);

  const [historicalVaR, setHistoricalVaR] = useState([
    { date: '2024-01', var95: -125000, var99: -185000, actualPnL: -95000 },
    { date: '2024-02', var95: -135000, var99: -195000, actualPnL: 45000 },
    { date: '2024-03', var95: -145000, var99: -205000, actualPnL: -125000 },
    { date: '2024-04', var95: -155000, var99: -215000, actualPnL: 85000 },
    { date: '2024-05', var95: -165000, var99: -225000, actualPnL: -35000 },
    { date: '2024-06', var95: -175000, var99: -235000, actualPnL: 125000 }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        totalValue: prev.totalValue + (Math.random() - 0.5) * 50000,
        sharpeRatio: prev.sharpeRatio + (Math.random() - 0.5) * 0.05,
        beta: prev.beta + (Math.random() - 0.5) * 0.02,
        alpha: prev.alpha + (Math.random() - 0.5) * 0.005,
        var95: prev.var95 + (Math.random() - 0.5) * 10000
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">PORTFOLIO ANALYTICS</h2>
          <div className="ml-auto text-xs text-terminal-muted">
            Quant Engine: <span className="text-terminal-green">ACTIVE</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Risk-Adjusted Performance Metrics */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">RISK-ADJUSTED PERFORMANCE</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-terminal-muted">Sharpe Ratio</div>
              <div className="text-lg font-bold text-terminal-green financial-number">
                {metrics.sharpeRatio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Sortino Ratio</div>
              <div className="text-lg font-bold text-terminal-cyan financial-number">
                {metrics.sortinoRatio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Calmar Ratio</div>
              <div className="text-lg font-bold text-terminal-amber financial-number">
                {metrics.calmarRatio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Alpha (Annualized)</div>
              <div className={`text-sm font-semibold financial-number ${
                metrics.alpha > 0 ? 'text-terminal-green' : 'text-terminal-red'
              }`}>
                {(metrics.alpha * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Information Ratio</div>
              <div className="text-sm font-semibold text-terminal-text financial-number">
                {metrics.informationRatio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Treynor Ratio</div>
              <div className="text-sm font-semibold text-terminal-text financial-number">
                {metrics.treynorRatio.toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">RISK METRICS</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-terminal-muted">VaR (95%, 1D)</div>
              <div className="text-sm font-semibold text-terminal-red financial-number">
                ${metrics.var95.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">CVaR (Expected Shortfall)</div>
              <div className="text-sm font-semibold text-terminal-red financial-number">
                ${metrics.cvar.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Max Drawdown</div>
              <div className="text-sm font-semibold text-terminal-red financial-number">
                {(metrics.maxDrawdown * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Tracking Error</div>
              <div className="text-sm font-semibold text-terminal-amber financial-number">
                {(metrics.trackingError * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>

        {/* Performance Attribution */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">FACTOR ATTRIBUTION</h3>
          <div className="space-y-2">
            {attribution.map((item, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded">
                <div className="flex-1">
                  <div className="text-xs font-medium text-terminal-text">{item.factor}</div>
                  <div className="text-xs text-terminal-muted">
                    Weight: {(item.weight * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-semibold financial-number ${
                    item.contribution > 0 ? 'text-terminal-green' : 'text-terminal-red'
                  }`}>
                    {item.contribution > 0 ? '+' : ''}{(item.contribution * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-terminal-muted">
                    Return: {(item.return * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Decomposition */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">RISK DECOMPOSITION</h3>
          <div className="h-32 mb-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskDecomp}>
                <XAxis 
                  dataKey="factor" 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <Bar dataKey="percentage" fill="#ff6b35" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* VaR Backtesting */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">VAR BACKTESTING</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historicalVaR}>
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
                  dataKey="var95" 
                  stroke="#ff4757" 
                  strokeWidth={2}
                  dot={false}
                  name="VaR 95%"
                />
                <Line 
                  type="monotone" 
                  dataKey="actualPnL" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                  name="Actual P&L"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
