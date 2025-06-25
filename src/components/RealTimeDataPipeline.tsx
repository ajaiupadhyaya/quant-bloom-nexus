
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { Activity, Zap, Database, AlertCircle, CheckCircle, Server } from 'lucide-react';

interface DataSource {
  name: string;
  status: 'connected' | 'disconnected' | 'error';
  latency: number;
  throughput: number;
  reliability: number;
}

interface DataQuality {
  source: string;
  anomalies: number;
  outliers: number;
  lastCheck: string;
  quality: number;
}

export const RealTimeDataPipeline = () => {
  const [dataSources, setDataSources] = useState<DataSource[]>([
    { name: 'Alpha Vantage', status: 'connected', latency: 15, throughput: 1250, reliability: 99.8 },
    { name: 'Polygon.io', status: 'connected', latency: 8, throughput: 2150, reliability: 99.9 },
    { name: 'IEX Cloud', status: 'connected', latency: 12, throughput: 1800, reliability: 99.7 },
    { name: 'Quandl', status: 'error', latency: 45, throughput: 0, reliability: 0 },
    { name: 'Redis Cache', status: 'connected', latency: 2, throughput: 15000, reliability: 100 }
  ]);

  const [dataQuality, setDataQuality] = useState<DataQuality[]>([
    { source: 'Equity Feed', anomalies: 2, outliers: 0, lastCheck: '10:32:15', quality: 99.8 },
    { source: 'Options Feed', anomalies: 0, outliers: 1, lastCheck: '10:32:12', quality: 99.9 },
    { source: 'Forex Feed', anomalies: 5, outliers: 2, lastCheck: '10:32:18', quality: 98.5 },
    { source: 'Crypto Feed', anomalies: 12, outliers: 8, lastCheck: '10:32:10', quality: 96.2 }
  ]);

  const [streamingMetrics, setStreamingMetrics] = useState([
    { time: '10:30', messages: 12500, latency: 8.5, errors: 0 },
    { time: '10:31', messages: 13200, latency: 9.2, errors: 1 },
    { time: '10:32', messages: 14100, latency: 7.8, errors: 0 },
    { time: '10:33', messages: 13800, latency: 8.1, errors: 0 },
    { time: '10:34', messages: 15200, latency: 9.5, errors: 2 }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setDataSources(prev => prev.map(source => ({
        ...source,
        latency: source.latency + (Math.random() - 0.5) * 2,
        throughput: source.status === 'connected' ? source.throughput + (Math.random() - 0.5) * 100 : 0
      })));

      setStreamingMetrics(prev => {
        const newMetric = {
          time: new Date().toLocaleTimeString().slice(0, 5),
          messages: 12000 + Math.floor(Math.random() * 5000),
          latency: 7 + Math.random() * 5,
          errors: Math.random() > 0.9 ? Math.floor(Math.random() * 3) : 0
        };
        return [...prev.slice(-4), newMetric];
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-terminal-green';
      case 'error': return 'text-terminal-red';
      default: return 'text-terminal-amber';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="w-3 h-3" />;
      case 'error': return <AlertCircle className="w-3 h-3" />;
      default: return <Activity className="w-3 h-3" />;
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">DATA PIPELINE</h2>
          </div>
          <div className="flex items-center space-x-2">
            <Server className="w-3 h-3 text-terminal-green" />
            <span className="text-xs text-terminal-green">KAFKA ACTIVE</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Data Sources Status */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">DATA SOURCES</h3>
          <div className="space-y-2">
            {dataSources.map((source, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded">
                <div className="flex items-center space-x-2">
                  <div className={getStatusColor(source.status)}>
                    {getStatusIcon(source.status)}
                  </div>
                  <span className="text-xs font-medium text-terminal-text">{source.name}</span>
                </div>
                <div className="flex space-x-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Latency</div>
                    <div className="text-terminal-cyan font-semibold financial-number">
                      {source.latency.toFixed(1)}ms
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Throughput</div>
                    <div className="text-terminal-green font-semibold financial-number">
                      {source.throughput.toLocaleString()}/s
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Uptime</div>
                    <div className="text-terminal-text font-semibold financial-number">
                      {source.reliability.toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Streaming Metrics Chart */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">STREAMING METRICS</h3>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={streamingMetrics}>
                <XAxis 
                  dataKey="time" 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333333',
                    borderRadius: '4px',
                    color: '#ffffff'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="messages" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                  yAxisId="left"
                />
                <Line 
                  type="monotone" 
                  dataKey="latency" 
                  stroke="#ff6b35" 
                  strokeWidth={2}
                  dot={false}
                  yAxisId="right"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Data Quality Monitor */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">DATA QUALITY</h3>
          <div className="space-y-2">
            {dataQuality.map((quality, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded">
                <div className="flex-1">
                  <div className="text-xs font-medium text-terminal-text">{quality.source}</div>
                  <div className="text-xs text-terminal-muted">
                    Last Check: {quality.lastCheck}
                  </div>
                </div>
                <div className="flex space-x-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Quality</div>
                    <div className={`font-semibold financial-number ${
                      quality.quality > 99 ? 'text-terminal-green' : 
                      quality.quality > 95 ? 'text-terminal-amber' : 'text-terminal-red'
                    }`}>
                      {quality.quality.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Anomalies</div>
                    <div className="text-terminal-red font-semibold financial-number">
                      {quality.anomalies}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Outliers</div>
                    <div className="text-terminal-amber font-semibold financial-number">
                      {quality.outliers}
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
