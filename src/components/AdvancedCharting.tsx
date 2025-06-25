
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, BarChart, Bar } from 'recharts';
import { TrendingUp, BarChart3, Activity } from 'lucide-react';

interface ChartData {
  time: string;
  price: number;
  volume: number;
}

export const AdvancedCharting = () => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [chartType, setChartType] = useState<'line' | 'bar'>('line');

  useEffect(() => {
    // Generate mock chart data
    const generateData = () => {
      const data = [];
      const basePrice = 180 + Math.random() * 20;
      
      for (let i = 0; i < 50; i++) {
        const time = new Date(Date.now() - (50 - i) * 60000);
        const price = basePrice + Math.sin(i / 5) * 5 + Math.random() * 2;
        data.push({
          time: time.toLocaleTimeString(),
          price: price,
          volume: Math.floor(Math.random() * 1000000),
        });
      }
      return data;
    };

    setChartData(generateData());
    
    const interval = setInterval(() => {
      setChartData(generateData());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="terminal-panel h-full flex flex-col">
      {/* Chart Header */}
      <div className="border-b border-terminal-border p-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">ADVANCED CHARTING</h2>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setChartType('line')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              chartType === 'line' 
                ? 'bg-terminal-orange text-terminal-bg' 
                : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-border'
            }`}
          >
            <TrendingUp className="w-3 h-3" />
          </button>
          <button
            onClick={() => setChartType('bar')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              chartType === 'bar' 
                ? 'bg-terminal-orange text-terminal-bg' 
                : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-border'
            }`}
          >
            <BarChart3 className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 p-3">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'line' ? (
            <LineChart data={chartData}>
              <XAxis 
                dataKey="time" 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: '#888888' }}
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: '#888888' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #333333',
                  borderRadius: '4px',
                  color: '#ffffff'
                }}
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
              />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#00d4ff" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#00d4ff' }}
              />
            </LineChart>
          ) : (
            <BarChart data={chartData}>
              <XAxis 
                dataKey="time" 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: '#888888' }}
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: '#888888' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #333333',
                  borderRadius: '4px',
                  color: '#ffffff'
                }}
                formatter={(value: number) => [value.toLocaleString(), 'Volume']}
              />
              <Bar dataKey="volume" fill="#ff6b35" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
      
      {/* Chart Info */}
      <div className="border-t border-terminal-border/50 p-2 flex justify-between text-xs text-terminal-muted">
        <span>Real-time Analysis</span>
        <span>Last Update: {new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
};
