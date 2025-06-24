
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { Satellite, TrendingUp, Globe, Zap, Activity, Brain } from 'lucide-react';

interface SatelliteData {
  location: string;
  activity: number;
  change: number;
  confidence: number;
  lastUpdate: string;
}

interface SocialSentiment {
  platform: string;
  mentions: number;
  sentiment: number;
  influence: number;
  trending: boolean;
}

interface WebTraffic {
  domain: string;
  visits: number;
  change: number;
  rank: number;
  category: string;
}

interface CreditCardData {
  sector: string;
  spending: number;
  change: number;
  confidence: number;
}

export const AlternativeData = () => {
  const [selectedDataType, setSelectedDataType] = useState('satellite');
  
  const [satelliteData, setSatelliteData] = useState<SatelliteData[]>([
    { location: 'Tesla Gigafactory', activity: 85, change: 12, confidence: 0.92, lastUpdate: '2 hours ago' },
    { location: 'Apple Park', activity: 78, change: -5, confidence: 0.89, lastUpdate: '1 hour ago' },
    { location: 'Amazon Warehouses', activity: 93, change: 8, confidence: 0.95, lastUpdate: '30 min ago' },
    { location: 'Walmart Stores', activity: 67, change: -2, confidence: 0.87, lastUpdate: '45 min ago' }
  ]);

  const [socialSentiment, setSocialSentiment] = useState<SocialSentiment[]>([
    { platform: 'Twitter', mentions: 45000, sentiment: 0.65, influence: 0.82, trending: true },
    { platform: 'Reddit', mentions: 12000, sentiment: 0.72, influence: 0.68, trending: false },
    { platform: 'LinkedIn', mentions: 8500, sentiment: 0.58, influence: 0.75, trending: true },
    { platform: 'News Media', mentions: 25000, sentiment: 0.45, influence: 0.92, trending: false }
  ]);

  const [webTraffic, setWebTraffic] = useState<WebTraffic[]>([
    { domain: 'apple.com', visits: 2500000, change: 15, rank: 12, category: 'Technology' },
    { domain: 'tesla.com', visits: 850000, change: 28, rank: 45, category: 'Automotive' },
    { domain: 'amazon.com', visits: 5200000, change: 5, rank: 3, category: 'E-commerce' },
    { domain: 'microsoft.com', visits: 1800000, change: -8, rank: 25, category: 'Technology' }
  ]);

  const [creditCardData, setCreditCardData] = useState<CreditCardData[]>([
    { sector: 'Retail', spending: 125000000, change: 8.5, confidence: 0.91 },
    { sector: 'Restaurants', spending: 85000000, change: 12.3, confidence: 0.88 },
    { sector: 'Travel', spending: 95000000, change: 22.1, confidence: 0.85 },
    { sector: 'Entertainment', spending: 65000000, change: -5.2, confidence: 0.83 }
  ]);

  const dataTypes = [
    { id: 'satellite', name: 'Satellite Imagery', icon: Satellite },
    { id: 'social', name: 'Social Sentiment', icon: Globe },
    { id: 'web', name: 'Web Traffic', icon: Activity },
    { id: 'credit', name: 'Credit Card', icon: TrendingUp }
  ];

  const renderDataContent = () => {
    switch (selectedDataType) {
      case 'satellite':
        return (
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-terminal-cyan mb-3">SATELLITE ACTIVITY ANALYSIS</h3>
            {satelliteData.map((data, index) => (
              <div key={index} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-terminal-text">{data.location}</div>
                  <div className="text-xs text-terminal-muted">{data.lastUpdate}</div>
                </div>
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Activity Level</div>
                    <div className="font-semibold text-terminal-cyan">{data.activity}%</div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">7D Change</div>
                    <div className={`font-semibold ${data.change > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                      {data.change > 0 ? '+' : ''}{data.change}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Confidence</div>
                    <div className="text-terminal-green font-semibold">{(data.confidence * 100).toFixed(0)}%</div>
                  </div>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-terminal-border rounded-full h-2">
                    <div 
                      className="bg-terminal-cyan h-2 rounded-full transition-all duration-500"
                      style={{ width: `${data.activity}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        );

      case 'social':
        return (
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-terminal-cyan mb-3">SOCIAL SENTIMENT TRACKING</h3>
            {socialSentiment.map((data, index) => (
              <div key={index} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-terminal-text">{data.platform}</span>
                    {data.trending && (
                      <span className="bg-terminal-orange/20 text-terminal-orange px-2 py-1 rounded text-xs">
                        TRENDING
                      </span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Mentions</div>
                    <div className="font-semibold text-terminal-cyan">{data.mentions.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Sentiment</div>
                    <div className={`font-semibold ${
                      data.sentiment > 0.6 ? 'text-terminal-green' : 
                      data.sentiment > 0.4 ? 'text-terminal-amber' : 'text-terminal-red'
                    }`}>
                      {(data.sentiment * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Influence</div>
                    <div className="text-terminal-green font-semibold">{(data.influence * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        );

      case 'web':
        return (
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-terminal-cyan mb-3">WEB TRAFFIC INTELLIGENCE</h3>
            {webTraffic.map((data, index) => (
              <div key={index} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-terminal-text">{data.domain}</div>
                  <div className="text-xs text-terminal-muted">Rank #{data.rank}</div>
                </div>
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Monthly Visits</div>
                    <div className="font-semibold text-terminal-cyan">
                      {(data.visits / 1000000).toFixed(1)}M
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">30D Change</div>
                    <div className={`font-semibold ${data.change > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                      {data.change > 0 ? '+' : ''}{data.change}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Category</div>
                    <div className="text-terminal-text font-semibold">{data.category}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        );

      case 'credit':
        return (
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-terminal-cyan mb-3">CREDIT CARD SPENDING DATA</h3>
            <div className="h-32 mb-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={creditCardData}>
                  <XAxis 
                    dataKey="sector" 
                    axisLine={false}
                    tick={{ fontSize: 10, fill: '#888888' }}
                  />
                  <YAxis 
                    axisLine={false}
                    tick={{ fontSize: 10, fill: '#888888' }}
                    tickFormatter={(value) => `${(value / 1000000).toFixed(0)}M`}
                  />
                  <Bar dataKey="spending" fill="#00d4ff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            {creditCardData.map((data, index) => (
              <div key={index} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-terminal-text">{data.sector}</div>
                  <div className="text-right">
                    <div className="text-sm font-semibold text-terminal-cyan">
                      ${(data.spending / 1000000).toFixed(0)}M
                    </div>
                    <div className={`text-xs font-semibold ${data.change > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                      {data.change > 0 ? '+' : ''}{data.change.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="mt-2 text-xs text-terminal-muted">
                  Confidence: <span className="text-terminal-green">{(data.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center space-x-2 mb-3">
          <Brain className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">ALTERNATIVE DATA</h2>
        </div>
        
        <div className="flex space-x-1">
          {dataTypes.map((type) => {
            const IconComponent = type.icon;
            return (
              <button
                key={type.id}
                onClick={() => setSelectedDataType(type.id)}
                className={`flex items-center space-x-1 px-2 py-1 text-xs rounded transition-colors ${
                  selectedDataType === type.id 
                    ? 'bg-terminal-orange text-terminal-bg' 
                    : 'bg-terminal-border text-terminal-muted hover:bg-terminal-orange/20'
                }`}
              >
                <IconComponent className="w-3 h-3" />
                <span>{type.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 p-3 overflow-y-auto">
        {renderDataContent()}
      </div>
    </div>
  );
};
