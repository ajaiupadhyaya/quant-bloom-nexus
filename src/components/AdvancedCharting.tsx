
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area, ComposedChart, Bar, ScatterChart, Scatter } from 'recharts';
import { BarChart3, TrendingUp, Volume2, Layers, Settings, ZoomIn, Move } from 'lucide-react';

interface ChartData {
  time: string;
  price: number;
  volume: number;
  vwap: number;
  volumeProfile: number;
  high: number;
  low: number;
  open: number;
  close: number;
}

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'bullish' | 'bearish' | 'neutral';
  period: number;
}

interface OrderFlowData {
  price: number;
  buyVolume: number;
  sellVolume: number;
  netFlow: number;
  size: number;
}

export const AdvancedCharting = () => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [timeframe, setTimeframe] = useState('1D');
  const [chartType, setChartType] = useState('candlestick');
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([
    { name: 'RSI(14)', value: 65.4, signal: 'bullish', period: 14 },
    { name: 'MACD', value: 2.45, signal: 'bullish', period: 12 },
    { name: 'Stoch', value: 72.1, signal: 'neutral', period: 14 },
    { name: 'BB%', value: 0.85, signal: 'bullish', period: 20 }
  ]);

  const [orderFlow, setOrderFlow] = useState<OrderFlowData[]>([]);
  const [volumeProfile, setVolumeProfile] = useState(true);
  const [marketMicrostructure, setMarketMicrostructure] = useState(true);

  useEffect(() => {
    const generateChartData = () => {
      const data = [];
      let basePrice = 180;
      
      for (let i = 0; i < 200; i++) {
        const timestamp = new Date(Date.now() - (200 - i) * 60000);
        const volatility = 0.02;
        const trend = Math.sin(i / 20) * 0.01;
        
        const open = basePrice;
        const change = (Math.random() - 0.5) * volatility + trend;
        const close = open + change;
        const high = Math.max(open, close) + Math.random() * 0.5;
        const low = Math.min(open, close) - Math.random() * 0.5;
        
        const volume = Math.floor(Math.random() * 1000000) + 500000;
        const vwap = (high + low + close) / 3;
        
        data.push({
          time: timestamp.toLocaleTimeString().slice(0, 5),
          price: close,
          volume: volume,
          vwap: vwap,
          volumeProfile: Math.floor(Math.random() * 50000) + 10000,
          high: high,
          low: low,
          open: open,
          close: close
        });
        
        basePrice = close;
      }
      return data;
    };

    const generateOrderFlow = () => {
      const data = [];
      const basePrice = 180;
      
      for (let i = 0; i < 20; i++) {
        const price = basePrice + (i - 10) * 0.25;
        const buyVol = Math.floor(Math.random() * 10000) + 1000;
        const sellVol = Math.floor(Math.random() * 10000) + 1000;
        
        data.push({
          price: price,
          buyVolume: buyVol,
          sellVolume: sellVol,
          netFlow: buyVol - sellVol,
          size: Math.abs(buyVol - sellVol)
        });
      }
      return data;
    };

    setChartData(generateChartData());
    setOrderFlow(generateOrderFlow());
    
    const interval = setInterval(() => {
      setChartData(generateChartData());
      setOrderFlow(generateOrderFlow());
      
      setIndicators(prev => prev.map(ind => ({
        ...ind,
        value: ind.value + (Math.random() - 0.5) * 5
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, [timeframe]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'bullish': return 'text-terminal-green';
      case 'bearish': return 'text-terminal-red';
      case 'neutral': return 'text-terminal-amber';
      default: return 'text-terminal-muted';
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">ADVANCED CHARTING</h2>
          </div>
          <div className="flex items-center space-x-2">
            <button className="terminal-button text-xs">
              <Layers className="w-3 h-3 mr-1" />
              Overlays
            </button>
            <button className="terminal-button text-xs">
              <Settings className="w-3 h-3 mr-1" />
              Settings
            </button>
          </div>
        </div>
        
        {/* Chart Controls */}
        <div className="flex items-center justify-between mt-2">
          <div className="flex space-x-2">
            {['1m', '5m', '15m', '1H', '4H', '1D'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  timeframe === tf 
                    ? 'bg-terminal-orange text-terminal-bg' 
                    : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-border'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
          
          <div className="flex space-x-2">
            {['line', 'candlestick', 'volume', 'heikin-ashi'].map((type) => (
              <button
                key={type}
                onClick={() => setChartType(type)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  chartType === type 
                    ? 'bg-terminal-cyan text-terminal-bg' 
                    : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-border'
                }`}
              >
                {type.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Main Chart Area */}
        <div className="flex-1 flex flex-col">
          {/* Price Chart */}
          <div className="flex-1 p-2">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData.slice(-50)}>
                <XAxis 
                  dataKey="time" 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                  domain={['dataMin - 1', 'dataMax + 1']}
                />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="vwap" 
                  stroke="#ff6b35" 
                  strokeWidth={1}
                  dot={false}
                  strokeDasharray="5 5"
                />
                {volumeProfile && (
                  <Bar 
                    dataKey="volumeProfile" 
                    fill="#333333" 
                    opacity={0.3}
                    yAxisId="volume"
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Volume Chart */}
          <div className="h-20 p-2 border-t border-terminal-border/50">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData.slice(-50)}>
                <XAxis 
                  dataKey="time" 
                  axisLine={false}
                  tick={{ fontSize: 9, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 9, fill: '#888888' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="volume" 
                  stroke="#00ff88" 
                  fill="#00ff88" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-48 border-l border-terminal-border/50 flex flex-col">
          {/* Technical Indicators */}
          <div className="p-2 border-b border-terminal-border/50">
            <h3 className="text-xs font-medium text-terminal-cyan mb-2">INDICATORS</h3>
            <div className="space-y-2">
              {indicators.map((indicator, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-xs text-terminal-text">{indicator.name}</span>
                  <div className="flex items-center space-x-1">
                    <span className={`text-xs font-semibold ${getSignalColor(indicator.signal)}`}>
                      {indicator.value.toFixed(1)}
                    </span>
                    <div className={`w-2 h-2 rounded-full ${
                      indicator.signal === 'bullish' ? 'bg-terminal-green' :
                      indicator.signal === 'bearish' ? 'bg-terminal-red' : 'bg-terminal-amber'
                    }`} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Order Flow */}
          {marketMicrostructure && (
            <div className="flex-1 p-2">
              <h3 className="text-xs font-medium text-terminal-cyan mb-2">ORDER FLOW</h3>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={orderFlow}>
                    <XAxis 
                      dataKey="netFlow" 
                      axisLine={false}
                      tick={{ fontSize: 8, fill: '#888888' }}
                    />
                    <YAxis 
                      dataKey="price" 
                      axisLine={false}
                      tick={{ fontSize: 8, fill: '#888888' }}
                    />
                    <Scatter 
                      dataKey="size" 
                      fill="#00d4ff" 
                      opacity={0.6}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-2 space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-terminal-muted">Buy Flow</span>
                  <span className="text-terminal-green">
                    {orderFlow.reduce((sum, item) => sum + Math.max(0, item.netFlow), 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-terminal-muted">Sell Flow</span>
                  <span className="text-terminal-red">
                    {Math.abs(orderFlow.reduce((sum, item) => sum + Math.min(0, item.netFlow), 0)).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
