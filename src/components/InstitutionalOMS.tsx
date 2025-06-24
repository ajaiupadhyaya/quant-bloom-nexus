
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Layers, Clock, Shield, Zap, Target, Activity } from 'lucide-react';

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: 'market' | 'limit' | 'iceberg' | 'twap' | 'vwap' | 'pov';
  quantity: number;
  filled: number;
  price?: number;
  avgPrice: number;
  status: 'pending' | 'working' | 'filled' | 'cancelled' | 'rejected';
  algorithm?: string;
  timestamp: Date;
  childOrders: number;
  participation: number;
}

interface ExecutionMetrics {
  symbol: string;
  vwap: number;
  implementation: number;
  marketImpact: number;
  timing: number;
  slippage: number;
}

export const InstitutionalOMS = () => {
  const [orders, setOrders] = useState<Order[]>([
    {
      id: 'ORD-001',
      symbol: 'AAPL',
      side: 'buy',
      orderType: 'twap',
      quantity: 100000,
      filled: 65000,
      avgPrice: 180.15,
      status: 'working',
      algorithm: 'TWAP-Aggressive',
      timestamp: new Date(),
      childOrders: 45,
      participation: 0.15
    },
    {
      id: 'ORD-002',
      symbol: 'MSFT',
      side: 'sell',
      orderType: 'vwap',
      quantity: 75000,
      filled: 75000,
      avgPrice: 340.85,
      status: 'filled',
      algorithm: 'VWAP-Neutral',
      timestamp: new Date(Date.now() - 3600000),
      childOrders: 32,
      participation: 0.12
    },
    {
      id: 'ORD-003',
      symbol: 'GOOGL',
      side: 'buy',
      orderType: 'pov',
      quantity: 25000,
      filled: 12500,
      avgPrice: 2751.20,
      status: 'working',
      algorithm: 'POV-Passive',
      timestamp: new Date(Date.now() - 1800000),
      childOrders: 18,
      participation: 0.08
    }
  ]);

  const [executionMetrics, setExecutionMetrics] = useState<ExecutionMetrics[]>([
    { symbol: 'AAPL', vwap: 180.25, implementation: -0.05, marketImpact: 0.03, timing: -0.02, slippage: 0.01 },
    { symbol: 'MSFT', vwap: 340.60, implementation: 0.08, marketImpact: -0.02, timing: 0.05, slippage: 0.05 },
    { symbol: 'GOOGL', vwap: 2750.80, implementation: -0.12, marketImpact: 0.08, timing: -0.15, slippage: 0.05 }
  ]);

  const [algorithmPerformance, setAlgorithmPerformance] = useState([
    { algorithm: 'TWAP', avgSlippage: 0.02, successRate: 0.94, volume: 15000000 },
    { algorithm: 'VWAP', avgSlippage: 0.015, successRate: 0.96, volume: 25000000 },
    { algorithm: 'POV', avgSlippage: 0.025, successRate: 0.91, volume: 8000000 },
    { algorithm: 'Implementation Shortfall', avgSlippage: 0.018, successRate: 0.93, volume: 18000000 }
  ]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-terminal-green';
      case 'working': return 'text-terminal-cyan';
      case 'cancelled': return 'text-terminal-amber';
      case 'rejected': return 'text-terminal-red';
      default: return 'text-terminal-muted';
    }
  };

  const getOrderTypeColor = (type: string) => {
    switch (type) {
      case 'twap': return 'bg-terminal-green/20 text-terminal-green';
      case 'vwap': return 'bg-terminal-cyan/20 text-terminal-cyan';
      case 'pov': return 'bg-terminal-amber/20 text-terminal-amber';
      case 'iceberg': return 'bg-terminal-orange/20 text-terminal-orange';
      default: return 'bg-terminal-border/20 text-terminal-muted';
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">INSTITUTIONAL OMS</h2>
          </div>
          <div className="flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-terminal-green rounded-full" />
              <span className="text-terminal-muted">Connected to Prime Broker</span>
            </div>
            <div className="text-terminal-cyan">
              Active Orders: {orders.filter(o => o.status === 'working').length}
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Active Orders */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">ACTIVE ORDERS</h3>
          <div className="space-y-2">
            {orders.map((order) => (
              <div key={order.id} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <span className="font-mono font-semibold text-terminal-cyan">{order.symbol}</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getOrderTypeColor(order.orderType)}`}>
                      {order.orderType.toUpperCase()}
                    </span>
                    <span className={`text-xs font-medium ${getStatusColor(order.status)}`}>
                      {order.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-xs text-terminal-muted">
                    {order.timestamp.toLocaleTimeString()}
                  </div>
                </div>
                
                <div className="grid grid-cols-4 gap-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Side / Quantity</div>
                    <div className={`font-semibold ${order.side === 'buy' ? 'text-terminal-green' : 'text-terminal-red'}`}>
                      {order.side.toUpperCase()} {order.quantity.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Filled / Avg Price</div>
                    <div className="font-semibold text-terminal-text">
                      {order.filled.toLocaleString()} @ ${order.avgPrice.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Algorithm</div>
                    <div className="font-semibold text-terminal-cyan">
                      {order.algorithm || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Child Orders</div>
                    <div className="font-semibold text-terminal-text">
                      {order.childOrders}
                    </div>
                  </div>
                </div>
                
                {/* Progress Bar */}
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-terminal-muted mb-1">
                    <span>Fill Progress</span>
                    <span>{((order.filled / order.quantity) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-terminal-border rounded-full h-2">
                    <div 
                      className="bg-terminal-cyan h-2 rounded-full transition-all duration-500"
                      style={{ width: `${(order.filled / order.quantity) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Execution Quality */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">EXECUTION QUALITY</h3>
          <div className="space-y-2">
            {executionMetrics.map((metric, index) => (
              <div key={index} className="bg-terminal-bg/30 p-2 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono font-semibold text-terminal-cyan">{metric.symbol}</span>
                  <span className="text-xs text-terminal-muted">
                    VWAP: ${metric.vwap.toFixed(2)}
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <div className="text-terminal-muted">Implementation</div>
                    <div className={`font-semibold financial-number ${
                      metric.implementation < 0 ? 'text-terminal-green' : 'text-terminal-red'
                    }`}>
                      {metric.implementation.toFixed(3)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Market Impact</div>
                    <div className={`font-semibold financial-number ${
                      Math.abs(metric.marketImpact) < 0.05 ? 'text-terminal-green' : 'text-terminal-amber'
                    }`}>
                      {metric.marketImpact.toFixed(3)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Timing</div>
                    <div className={`font-semibold financial-number ${
                      metric.timing < 0 ? 'text-terminal-green' : 'text-terminal-red'
                    }`}>
                      {metric.timing.toFixed(3)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Slippage</div>
                    <div className={`font-semibold financial-number ${
                      metric.slippage < 0.02 ? 'text-terminal-green' : 'text-terminal-amber'
                    }`}>
                      {metric.slippage.toFixed(3)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Algorithm Performance */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">ALGORITHM PERFORMANCE</h3>
          <div className="h-32 mb-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={algorithmPerformance}>
                <XAxis 
                  dataKey="algorithm" 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                  tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                />
                <Bar dataKey="avgSlippage" fill="#ff6b35" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="space-y-2">
            {algorithmPerformance.map((algo, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded text-xs">
                <div className="font-semibold text-terminal-text">{algo.algorithm}</div>
                <div className="flex space-x-4">
                  <div>
                    <span className="text-terminal-muted">Success Rate: </span>
                    <span className="text-terminal-green font-semibold">
                      {(algo.successRate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-terminal-muted">Volume: </span>
                    <span className="text-terminal-cyan font-semibold">
                      ${(algo.volume / 1000000).toFixed(1)}M
                    </span>
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
