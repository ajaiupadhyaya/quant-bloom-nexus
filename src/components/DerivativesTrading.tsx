
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Target, Zap, TrendingUp, Calculator, Activity, Layers } from 'lucide-react';

interface OptionChain {
  strike: number;
  callPrice: number;
  putPrice: number;
  callIV: number;
  putIV: number;
  callDelta: number;
  putDelta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  volume: number;
  openInterest: number;
}

interface BlackScholesInputs {
  spot: number;
  strike: number;
  timeToExpiry: number;
  riskFreeRate: number;
  volatility: number;
  dividendYield: number;
}

export const DerivativesTrading = () => {
  const [selectedExpiry, setSelectedExpiry] = useState('2024-07-19');
  const [underlyingPrice, setUnderlyingPrice] = useState(180.25);
  const [bsInputs, setBsInputs] = useState<BlackScholesInputs>({
    spot: 180.25,
    strike: 180,
    timeToExpiry: 0.0548, // 20 days
    riskFreeRate: 0.0525,
    volatility: 0.25,
    dividendYield: 0.015
  });

  const [optionChain, setOptionChain] = useState<OptionChain[]>([
    { strike: 170, callPrice: 12.45, putPrice: 1.85, callIV: 0.22, putIV: 0.24, callDelta: 0.85, putDelta: -0.15, gamma: 0.015, theta: -0.08, vega: 0.12, rho: 0.045, volume: 850, openInterest: 2450 },
    { strike: 175, callPrice: 8.20, putPrice: 2.95, callIV: 0.24, putIV: 0.25, callDelta: 0.72, putDelta: -0.28, gamma: 0.025, theta: -0.12, vega: 0.18, rho: 0.038, volume: 1250, openInterest: 3200 },
    { strike: 180, callPrice: 4.85, putPrice: 4.60, callIV: 0.26, putIV: 0.26, callDelta: 0.52, putDelta: -0.48, gamma: 0.035, theta: -0.15, vega: 0.22, rho: 0.028, volume: 2850, openInterest: 5600 },
    { strike: 185, callPrice: 2.40, putPrice: 7.15, callIV: 0.27, putIV: 0.28, callDelta: 0.32, putDelta: -0.68, gamma: 0.028, theta: -0.12, vega: 0.19, rho: 0.018, volume: 1850, openInterest: 4100 },
    { strike: 190, callPrice: 1.15, putPrice: 10.90, callIV: 0.29, putIV: 0.31, callDelta: 0.18, putDelta: -0.82, gamma: 0.018, theta: -0.08, vega: 0.14, rho: 0.012, volume: 950, openInterest: 2800 }
  ]);

  const [volatilitySurface, setVolatilitySurface] = useState([
    { strike: 170, expiry1: 0.22, expiry2: 0.24, expiry3: 0.26 },
    { strike: 175, expiry1: 0.24, expiry2: 0.25, expiry3: 0.27 },
    { strike: 180, expiry1: 0.26, expiry2: 0.26, expiry3: 0.28 },
    { strike: 185, expiry1: 0.27, expiry2: 0.28, expiry3: 0.29 },
    { strike: 190, expiry1: 0.29, expiry2: 0.31, expiry3: 0.32 }
  ]);

  // Black-Scholes calculation (simplified)
  const calculateBlackScholes = (inputs: BlackScholesInputs, isCall: boolean = true) => {
    const { spot, strike, timeToExpiry, riskFreeRate, volatility, dividendYield } = inputs;
    const d1 = (Math.log(spot / strike) + (riskFreeRate - dividendYield + (volatility ** 2) / 2) * timeToExpiry) / (volatility * Math.sqrt(timeToExpiry));
    const d2 = d1 - volatility * Math.sqrt(timeToExpiry);
    
    // Simplified normal CDF approximation
    const normalCDF = (x: number) => 0.5 * (1 + Math.sign(x) * Math.sqrt(1 - Math.exp(-2 * x * x / Math.PI)));
    
    if (isCall) {
      return spot * Math.exp(-dividendYield * timeToExpiry) * normalCDF(d1) - strike * Math.exp(-riskFreeRate * timeToExpiry) * normalCDF(d2);
    } else {
      return strike * Math.exp(-riskFreeRate * timeToExpiry) * normalCDF(-d2) - spot * Math.exp(-dividendYield * timeToExpiry) * normalCDF(-d1);
    }
  };

  const expiries = ['2024-07-19', '2024-08-16', '2024-09-20', '2024-10-18'];

  useEffect(() => {
    const interval = setInterval(() => {
      setUnderlyingPrice(prev => prev + (Math.random() - 0.5) * 0.5);
      setBsInputs(prev => ({ ...prev, spot: underlyingPrice }));
    }, 2000);

    return () => clearInterval(interval);
  }, [underlyingPrice]);

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">DERIVATIVES TRADING</h2>
          </div>
          <div className="text-xs text-terminal-muted">
            Underlying: <span className="text-terminal-cyan financial-number">${underlyingPrice.toFixed(2)}</span>
          </div>
        </div>
        
        <div className="flex space-x-2 mt-2">
          {expiries.map((expiry) => (
            <button
              key={expiry}
              onClick={() => setSelectedExpiry(expiry)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                selectedExpiry === expiry 
                  ? 'bg-terminal-orange text-terminal-bg' 
                  : 'bg-terminal-border text-terminal-muted hover:bg-terminal-orange/20'
              }`}
            >
              {expiry}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Black-Scholes Calculator */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3 flex items-center">
            <Calculator className="w-3 h-3 mr-1" />
            BLACK-SCHOLES CALCULATOR
          </h3>
          <div className="grid grid-cols-3 gap-2 mb-3">
            <div>
              <label className="text-xs text-terminal-muted">Strike</label>
              <input
                type="number"
                value={bsInputs.strike}
                onChange={(e) => setBsInputs(prev => ({ ...prev, strike: Number(e.target.value) }))}
                className="terminal-input w-full text-xs"
              />
            </div>
            <div>
              <label className="text-xs text-terminal-muted">Vol (%)</label>
              <input
                type="number"
                value={(bsInputs.volatility * 100).toFixed(1)}
                onChange={(e) => setBsInputs(prev => ({ ...prev, volatility: Number(e.target.value) / 100 }))}
                className="terminal-input w-full text-xs"
              />
            </div>
            <div>
              <label className="text-xs text-terminal-muted">Days</label>
              <input
                type="number"
                value={(bsInputs.timeToExpiry * 365).toFixed(0)}
                onChange={(e) => setBsInputs(prev => ({ ...prev, timeToExpiry: Number(e.target.value) / 365 }))}
                className="terminal-input w-full text-xs"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 bg-terminal-bg/30 p-2 rounded">
            <div>
              <div className="text-xs text-terminal-muted">Call Price</div>
              <div className="text-sm font-semibold text-terminal-green financial-number">
                ${calculateBlackScholes(bsInputs, true).toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Put Price</div>
              <div className="text-sm font-semibold text-terminal-red financial-number">
                ${calculateBlackScholes(bsInputs, false).toFixed(2)}
              </div>
            </div>
          </div>
        </div>

        {/* Option Chain */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">OPTION CHAIN</h3>
          <div className="overflow-x-auto">
            <table className="data-table text-xs">
              <thead>
                <tr>
                  <th>Strike</th>
                  <th>Call</th>
                  <th>IV</th>
                  <th>Delta</th>
                  <th>Put</th>
                  <th>IV</th>
                  <th>Delta</th>
                  <th>Gamma</th>
                  <th>Theta</th>
                  <th>Vega</th>
                </tr>
              </thead>
              <tbody>
                {optionChain.map((option) => (
                  <tr key={option.strike} className={option.strike === 180 ? 'bg-terminal-border/20' : ''}>
                    <td className="font-mono font-semibold">${option.strike}</td>
                    <td className="text-terminal-green financial-number">${option.callPrice.toFixed(2)}</td>
                    <td className="financial-number">{(option.callIV * 100).toFixed(1)}%</td>
                    <td className="financial-number">{option.callDelta.toFixed(2)}</td>
                    <td className="text-terminal-red financial-number">${option.putPrice.toFixed(2)}</td>
                    <td className="financial-number">{(option.putIV * 100).toFixed(1)}%</td>
                    <td className="financial-number">{option.putDelta.toFixed(2)}</td>
                    <td className="financial-number">{option.gamma.toFixed(3)}</td>
                    <td className="text-terminal-red financial-number">{option.theta.toFixed(2)}</td>
                    <td className="financial-number">{option.vega.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Volatility Surface */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">VOLATILITY SURFACE</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={volatilitySurface}>
                <XAxis 
                  dataKey="strike" 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Line 
                  type="monotone" 
                  dataKey="expiry1" 
                  stroke="#00ff88" 
                  strokeWidth={2}
                  dot={false}
                  name="1M"
                />
                <Line 
                  type="monotone" 
                  dataKey="expiry2" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                  name="2M"
                />
                <Line 
                  type="monotone" 
                  dataKey="expiry3" 
                  stroke="#ff6b35" 
                  strokeWidth={2}
                  dot={false}
                  name="3M"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
