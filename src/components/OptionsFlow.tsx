import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Circle, Target } from 'lucide-react';
import { D3BarChart } from './D3BarChart';
import { D3PieChart } from './D3PieChart';

interface OptionsFlowData {
  strike: number;
  calls: number;
  puts: number;
  callVolume: number;
  putVolume: number;
  oi: number;
  type: 'calls' | 'puts';
}

interface OptionsFlowProps {
  symbol: string;
}

export const OptionsFlow = ({ symbol }: OptionsFlowProps) => {
  const [flowData, setFlowData] = useState<OptionsFlowData[]>([]);
  const [putCallRatio, setPutCallRatio] = useState(0.85);
  const [unusualActivity, setUnusualActivity] = useState([
    { strike: 180, type: 'calls', volume: 15000, premium: 2.45, unusual: true },
    { strike: 175, type: 'puts', volume: 12000, premium: 1.85, unusual: true },
    { strike: 185, type: 'calls', volume: 8500, premium: 1.25, unusual: false },
  ]);

  useEffect(() => {
    const generateFlowData = () => {
      const data = [];
      const baseStrike = 180;
      
      for (let i = -5; i <= 5; i++) {
        const strike = baseStrike + i * 2.5;
        data.push({
          strike,
          calls: Math.floor(Math.random() * 5000) + 1000,
          puts: Math.floor(Math.random() * 3000) + 500,
          callVolume: Math.floor(Math.random() * 10000) + 2000,
          putVolume: Math.floor(Math.random() * 8000) + 1500,
          oi: Math.floor(Math.random() * 15000) + 5000,
          type: Math.random() > 0.5 ? 'calls' : 'puts'
        });
      }
      return data;
    };

    setFlowData(generateFlowData());
    
    const interval = setInterval(() => {
      setFlowData(generateFlowData());
      setPutCallRatio(0.7 + Math.random() * 0.6);
    }, 4000);

    return () => clearInterval(interval);
  }, [symbol]);

  const pieData = [
    { name: 'Calls', value: 60, color: '#00ff88' },
    { name: 'Puts', value: 40, color: '#ff4757' }
  ];

  // Format data for D3 charts
  const volumeFlowData = flowData.slice(0, 6).map(d => ({
    x: d.strike.toString(),
    y: d.callVolume + d.putVolume,
    color: d.callVolume > d.putVolume ? '#00ff88' : '#ff4757'
  }));

  const distributionData = pieData.map(d => ({
    label: d.name,
    value: d.value,
    color: d.color
  }));

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">OPTIONS FLOW</h2>
          </div>
          <div className="text-xs text-terminal-muted">{symbol}</div>
        </div>
        
        {/* Put/Call Ratio */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-terminal-muted">P/C Ratio</span>
          <span className={`text-sm font-semibold ${
            putCallRatio > 1 ? 'text-terminal-red' : 'text-terminal-green'
          }`}>
            {putCallRatio.toFixed(2)}
          </span>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        {/* Unusual Activity */}
        <div className="p-2 border-b border-terminal-border/50">
          <div className="text-xs text-terminal-muted mb-2">UNUSUAL ACTIVITY</div>
          <div className="space-y-1">
            {unusualActivity.slice(0, 2).map((activity, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <Circle className={`w-2 h-2 ${activity.unusual ? 'text-terminal-orange' : 'text-terminal-muted'}`} />
                  <span>${activity.strike}</span>
                  <span className={activity.type === 'calls' ? 'text-terminal-green' : 'text-terminal-red'}>
                    {activity.type.toUpperCase()}
                  </span>
                </div>
                <div className="text-terminal-cyan financial-number">
                  {activity.volume.toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Options Flow Chart */}
        <div className="flex-1 p-2">
          <div className="grid grid-cols-2 gap-2 h-full">
            {/* Volume Flow */}
            <div>
              <div className="text-xs text-terminal-muted mb-1">Volume Flow</div>
              <div className="h-[80%]">
                <D3BarChart
                  data={volumeFlowData}
                  width={300}
                  height={120}
                  title="Volume Flow"
                  xLabel="Strike"
                  yLabel="Volume"
                />
              </div>
            </div>
            
            {/* Distribution */}
            <div>
              <div className="text-xs text-terminal-muted mb-1">Distribution</div>
              <div className="h-[80%]">
                <D3PieChart
                  data={distributionData}
                  width={300}
                  height={120}
                  title="Distribution"
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Strike Levels */}
        <div className="p-2 border-t border-terminal-border/50">
          <div className="text-xs text-terminal-muted mb-1">KEY STRIKES</div>
          <div className="grid grid-cols-3 gap-1 text-xs">
            <div className="text-center">
              <div className="text-terminal-red">175 PUT</div>
              <div className="text-terminal-muted">12K</div>
            </div>
            <div className="text-center">
              <div className="text-terminal-cyan">180 ATM</div>
              <div className="text-terminal-muted">25K</div>
            </div>
            <div className="text-center">
              <div className="text-terminal-green">185 CALL</div>
              <div className="text-terminal-muted">18K</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
