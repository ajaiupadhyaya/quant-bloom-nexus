import React, { useState, useEffect, useCallback } from 'react';
import { 
  Brain, Activity, TrendingUp, AlertTriangle, Settings, 
  BarChart3, LineChart, PieChart, Monitor, Shield, 
  Network, Database, Cpu, Zap, Globe, Target,
  BookOpen, Search, Filter, Calendar, Bell,
  ArrowUpDown, DollarSign, Percent, Timer
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useTradingStore, useMarketData, usePositions, useActivePortfolio } from '../lib/store/TradingStore';

// Bloomberg-style terminal colors
const terminalTheme = {
  bg: 'bg-black',
  panel: 'bg-gray-900',
  border: 'border-orange-500',
  text: 'text-orange-400',
  accent: 'text-cyan-400',
  success: 'text-green-400',
  warning: 'text-yellow-400',
  danger: 'text-red-400',
  muted: 'text-gray-500'
};

interface MarketOverviewData {
  index: string;
  value: number;
  change: number;
  changePercent: number;
}

interface EconomicEvent {
  time: string;
  event: string;
  currency: string;
  importance: 'low' | 'medium' | 'high';
  actual?: number;
  forecast?: number;
  previous?: number;
}

export const InstitutionalTradingTerminal: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [selectedTab, setSelectedTab] = useState('trading');
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connected');
  
  const { 
    selectedSymbol, 
    setSelectedSymbol,
    connectionStatus: storeConnectionStatus 
  } = useTradingStore();
  
  const marketData = useMarketData(selectedSymbol);
  const positions = usePositions();
  const activePortfolio = useActivePortfolio();

  // Sample market overview data
  const [marketOverview] = useState<MarketOverviewData[]>([
    { index: 'S&P 500', value: 4521.23, change: 12.45, changePercent: 0.28 },
    { index: 'NASDAQ', value: 15245.67, change: -8.23, changePercent: -0.05 },
    { index: 'DOW JONES', value: 34567.89, change: 156.78, changePercent: 0.45 },
    { index: 'RUSSELL 2000', value: 2123.45, change: 5.67, changePercent: 0.27 },
    { index: 'VIX', value: 18.45, change: -1.23, changePercent: -6.25 },
    { index: 'DXY', value: 104.67, change: 0.34, changePercent: 0.33 }
  ]);

  // Sample economic events
  const [economicEvents] = useState<EconomicEvent[]>([
    { time: '08:30', event: 'Non-Farm Payrolls', currency: 'USD', importance: 'high', forecast: 200, previous: 187 },
    { time: '10:00', event: 'ISM Manufacturing PMI', currency: 'USD', importance: 'medium', forecast: 49.2, previous: 48.7 },
    { time: '14:00', event: 'FOMC Meeting Minutes', currency: 'USD', importance: 'high' },
    { time: '15:30', event: 'Crude Oil Inventories', currency: 'USD', importance: 'medium', previous: -2.5 }
  ]);

  // Performance metrics
  const [systemMetrics] = useState({
    latency: 0.8,
    throughput: 125000,
    cpuUsage: 23,
    memoryUsage: 67,
    networkLatency: 2.3,
    dataProcessed: 2.4e6
  });

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setConnectionStatus(storeConnectionStatus);
  }, [storeConnectionStatus]);

  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
  }, [setSelectedSymbol]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className={`min-h-screen ${terminalTheme.bg} ${terminalTheme.text} font-mono overflow-hidden`}>
      {/* Terminal Header */}
      <div className={`h-16 ${terminalTheme.panel} border-b-2 ${terminalTheme.border} flex items-center justify-between px-6`}>
        <div className="flex items-center space-x-6">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Monitor className={`w-8 h-8 ${terminalTheme.text}`} />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
            </div>
            <div>
              <h1 className={`text-xl font-bold ${terminalTheme.text} bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent`}>
                INSTITUTIONAL TERMINAL
              </h1>
              <div className="text-xs text-gray-400">Bloomberg Terminal-Grade Platform v3.2.1</div>
            </div>
          </div>

          {/* Market Status */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span className={`text-sm ${connectionStatus === 'connected' ? terminalTheme.success : terminalTheme.danger}`}>
                {connectionStatus.toUpperCase()}
              </span>
            </div>
            <div className="text-sm text-gray-400">
              {currentTime.toLocaleTimeString()} EST
            </div>
          </div>

          {/* Quick Stats */}
          <div className="flex items-center space-x-6 text-sm">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span className="text-gray-400">Latency:</span>
              <span className={terminalTheme.accent}>{systemMetrics.latency}ms</span>
            </div>
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-gray-400">Throughput:</span>
              <span className={terminalTheme.accent}>{formatNumber(systemMetrics.throughput)}/s</span>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4" />
            <Progress value={systemMetrics.cpuUsage} className="w-16 h-2" />
            <span className="text-xs">{systemMetrics.cpuUsage}%</span>
          </div>
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4" />
            <Progress value={systemMetrics.memoryUsage} className="w-16 h-2" />
            <span className="text-xs">{systemMetrics.memoryUsage}%</span>
          </div>
          <Button variant="outline" size="sm" className="border-orange-500 text-orange-400 hover:bg-orange-500/10">
            <Settings className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Market Overview Ticker */}
      <div className={`h-12 ${terminalTheme.panel} border-b border-gray-700 flex items-center px-6 overflow-hidden`}>
        <div className="flex animate-marquee space-x-8">
          {marketOverview.map((item) => (
            <div key={item.index} className="flex items-center space-x-3 whitespace-nowrap">
              <span className="font-semibold text-gray-300">{item.index}</span>
              <span className={terminalTheme.accent}>{formatNumber(item.value)}</span>
              <span className={`${item.change >= 0 ? terminalTheme.success : terminalTheme.danger}`}>
                {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}
              </span>
              <span className={`text-xs ${item.changePercent >= 0 ? terminalTheme.success : terminalTheme.danger}`}>
                ({formatPercent(item.changePercent)})
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Terminal Interface */}
      <div className="h-[calc(100vh-112px)] flex">
        {/* Left Sidebar - Navigation & Watchlist */}
        <div className={`w-80 ${terminalTheme.panel} border-r border-gray-700 flex flex-col`}>
          <Tabs value={selectedTab} onValueChange={setSelectedTab} className="flex-1 flex flex-col">
            <TabsList className="grid w-full grid-cols-4 bg-gray-800 m-2">
              <TabsTrigger value="trading" className="text-xs">Trading</TabsTrigger>
              <TabsTrigger value="analysis" className="text-xs">Analysis</TabsTrigger>
              <TabsTrigger value="research" className="text-xs">Research</TabsTrigger>
              <TabsTrigger value="risk" className="text-xs">Risk</TabsTrigger>
            </TabsList>

            <TabsContent value="trading" className="flex-1 px-2">
              <Card className="bg-gray-800 border-gray-700 mb-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <Target className="w-4 h-4 mr-2" />
                    Quick Trading
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex space-x-2">
                    <Button size="sm" className="flex-1 bg-green-600 hover:bg-green-700">BUY</Button>
                    <Button size="sm" className="flex-1 bg-red-600 hover:bg-red-700">SELL</Button>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <label className="text-gray-400">Quantity</label>
                      <input className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" defaultValue="100" />
                    </div>
                    <div>
                      <label className="text-gray-400">Price</label>
                      <input className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white" defaultValue="150.25" />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400">Watchlist</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-64">
                    {['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'].map((symbol) => (
                      <div 
                        key={symbol}
                        className={`p-3 border-b border-gray-700 cursor-pointer hover:bg-gray-700 ${
                          selectedSymbol === symbol ? 'bg-gray-700' : ''
                        }`}
                        onClick={() => handleSymbolChange(symbol)}
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-semibold text-cyan-400">{symbol}</span>
                          <span className="text-green-400">$150.25</span>
                        </div>
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>+2.45</span>
                          <span>+1.65%</span>
                        </div>
                      </div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analysis" className="flex-1 px-2">
              <Card className="bg-gray-800 border-gray-700 mb-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <Brain className="w-4 h-4 mr-2" />
                    AI Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Price Prediction</span>
                      <span className="text-green-400">↗ $158.50</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confidence</span>
                      <span className="text-yellow-400">87%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sentiment</span>
                      <span className="text-green-400">Bullish</span>
                    </div>
                  </div>
                  <Progress value={87} className="h-2" />
                </CardContent>
              </Card>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400">Technical Indicators</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {[
                    { name: 'RSI', value: 67.3, signal: 'Neutral' },
                    { name: 'MACD', value: 1.23, signal: 'Bullish' },
                    { name: 'SMA(20)', value: 148.56, signal: 'Above' },
                    { name: 'BB', value: 152.4, signal: 'Upper' }
                  ].map((indicator) => (
                    <div key={indicator.name} className="flex justify-between text-xs">
                      <span className="text-gray-400">{indicator.name}</span>
                      <span className="text-cyan-400">{indicator.value}</span>
                      <span className={`${
                        indicator.signal === 'Bullish' || indicator.signal === 'Above' ? 'text-green-400' : 
                        indicator.signal === 'Bearish' || indicator.signal === 'Below' ? 'text-red-400' : 
                        'text-yellow-400'
                      }`}>
                        {indicator.signal}
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="research" className="flex-1 px-2">
              <Card className="bg-gray-800 border-gray-700 mb-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <Calendar className="w-4 h-4 mr-2" />
                    Economic Calendar
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-48">
                    {economicEvents.map((event, index) => (
                      <div key={index} className="p-2 border-b border-gray-700">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <div className="text-xs font-semibold text-cyan-400">{event.time}</div>
                            <div className="text-xs text-gray-300">{event.event}</div>
                          </div>
                          <Badge 
                            variant="outline" 
                            className={`text-xs ${
                              event.importance === 'high' ? 'border-red-400 text-red-400' :
                              event.importance === 'medium' ? 'border-yellow-400 text-yellow-400' :
                              'border-gray-400 text-gray-400'
                            }`}
                          >
                            {event.importance.toUpperCase()}
                          </Badge>
                        </div>
                        {event.forecast && (
                          <div className="text-xs text-gray-400 mt-1">
                            F: {event.forecast} | P: {event.previous}
                          </div>
                        )}
                      </div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400">News Flow</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-32">
                    {[
                      { time: '14:23', headline: 'Fed Chair Powell signals rate cuts ahead', source: 'Reuters' },
                      { time: '13:45', headline: 'Tech earnings beat expectations', source: 'Bloomberg' },
                      { time: '13:12', headline: 'Oil prices surge on supply concerns', source: 'WSJ' }
                    ].map((news, index) => (
                      <div key={index} className="p-2 border-b border-gray-700">
                        <div className="text-xs text-gray-400">{news.time} | {news.source}</div>
                        <div className="text-xs text-gray-300">{news.headline}</div>
                      </div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="risk" className="flex-1 px-2">
              <Card className="bg-gray-800 border-gray-700 mb-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <Shield className="w-4 h-4 mr-2" />
                    Risk Monitor
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-xs space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Portfolio VaR (1D)</span>
                      <span className="text-red-400">-$12,450</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Max Drawdown</span>
                      <span className="text-yellow-400">-3.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sharpe Ratio</span>
                      <span className="text-green-400">1.67</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Beta</span>
                      <span className="text-cyan-400">0.85</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400">Risk Alerts</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-32">
                    {[
                      { type: 'warning', message: 'Position concentration exceeds 15%', symbol: 'AAPL' },
                      { type: 'info', message: 'Volatility spike detected', symbol: 'TSLA' },
                      { type: 'error', message: 'Stop loss triggered', symbol: 'NVDA' }
                    ].map((alert, index) => (
                      <div key={index} className="p-2 border-b border-gray-700 flex items-center space-x-2">
                        <AlertTriangle className={`w-3 h-3 ${
                          alert.type === 'error' ? 'text-red-400' :
                          alert.type === 'warning' ? 'text-yellow-400' :
                          'text-blue-400'
                        }`} />
                        <div className="flex-1">
                          <div className="text-xs text-gray-300">{alert.message}</div>
                          <div className="text-xs text-gray-500">{alert.symbol}</div>
                        </div>
                      </div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Top Row - Charts and Analysis */}
          <div className="flex-1 flex">
            {/* Price Chart */}
            <div className="flex-1 p-4">
              <Card className="h-full bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-center">
                    <CardTitle className="text-lg text-orange-400">{selectedSymbol}</CardTitle>
                    <div className="flex items-center space-x-4 text-sm">
                      <span className="text-cyan-400">$150.25</span>
                      <span className="text-green-400">+2.45 (+1.65%)</span>
                      <Badge variant="outline" className="border-green-400 text-green-400">
                        MARKET OPEN
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="h-[calc(100%-80px)]">
                  <div className="w-full h-full bg-gray-900 rounded border border-gray-600 flex items-center justify-center">
                    <div className="text-center">
                      <LineChart className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                      <p className="text-gray-500">Real-time Price Chart</p>
                      <p className="text-xs text-gray-600">Candlestick, Volume, Technical Indicators</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Order Book & Level II */}
            <div className="w-80 p-4">
              <Card className="h-full bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400">Level II - {selectedSymbol}</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="grid grid-cols-3 gap-1 p-2 text-xs text-gray-400 border-b border-gray-700">
                    <span>Size</span>
                    <span className="text-center">Price</span>
                    <span className="text-right">Size</span>
                  </div>
                  <ScrollArea className="h-64">
                    {/* Asks */}
                    {[
                      { price: 150.28, bidSize: 0, askSize: 500 },
                      { price: 150.27, bidSize: 0, askSize: 750 },
                      { price: 150.26, bidSize: 0, askSize: 300 },
                      { price: 150.25, bidSize: 0, askSize: 1200 },
                      { price: 150.24, bidSize: 0, askSize: 800 }
                    ].map((level, index) => (
                      <div key={`ask-${index}`} className="grid grid-cols-3 gap-1 p-1 text-xs hover:bg-gray-700">
                        <span className="text-gray-500"></span>
                        <span className="text-center text-red-400">${level.price.toFixed(2)}</span>
                        <span className="text-right text-gray-300">{formatNumber(level.askSize)}</span>
                      </div>
                    ))}
                    
                    {/* Spread */}
                    <div className="bg-gray-700 p-1 text-center text-xs">
                      <span className="text-yellow-400">Spread: $0.02</span>
                    </div>

                    {/* Bids */}
                    {[
                      { price: 150.23, bidSize: 900, askSize: 0 },
                      { price: 150.22, bidSize: 1100, askSize: 0 },
                      { price: 150.21, bidSize: 600, askSize: 0 },
                      { price: 150.20, bidSize: 400, askSize: 0 },
                      { price: 150.19, bidSize: 800, askSize: 0 }
                    ].map((level, index) => (
                      <div key={`bid-${index}`} className="grid grid-cols-3 gap-1 p-1 text-xs hover:bg-gray-700">
                        <span className="text-gray-300">{formatNumber(level.bidSize)}</span>
                        <span className="text-center text-green-400">${level.price.toFixed(2)}</span>
                        <span className="text-gray-500"></span>
                      </div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Bottom Row - Portfolio, Orders, News */}
          <div className="h-80 flex border-t border-gray-700">
            {/* Portfolio Summary */}
            <div className="flex-1 p-4">
              <Card className="h-full bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <PieChart className="w-4 h-4 mr-2" />
                    Portfolio Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="text-xs text-gray-400">Total Value</div>
                      <div className="text-lg font-semibold text-cyan-400">{formatCurrency(1250000)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Day P&L</div>
                      <div className="text-lg font-semibold text-green-400">+$12,450</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Cash Balance</div>
                      <div className="text-sm text-gray-300">{formatCurrency(125000)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Margin Used</div>
                      <div className="text-sm text-gray-300">{formatCurrency(75000)}</div>
                    </div>
                  </div>
                  
                  <ScrollArea className="h-32">
                    <div className="space-y-2">
                      {positions.map((position, index) => (
                        <div key={index} className="flex justify-between items-center text-xs border-b border-gray-700 pb-1">
                          <span className="text-cyan-400">{position.symbol}</span>
                          <span className="text-gray-300">{position.quantity}</span>
                          <span className={position.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {formatCurrency(position.unrealizedPnL)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            {/* Order Management */}
            <div className="flex-1 p-4">
              <Card className="h-full bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <ArrowUpDown className="w-4 h-4 mr-2" />
                    Order Management
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <Tabs defaultValue="orders" className="h-full">
                    <TabsList className="grid w-full grid-cols-3 bg-gray-700 m-2">
                      <TabsTrigger value="orders" className="text-xs">Orders</TabsTrigger>
                      <TabsTrigger value="fills" className="text-xs">Fills</TabsTrigger>
                      <TabsTrigger value="blotter" className="text-xs">Blotter</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="orders" className="px-2">
                      <ScrollArea className="h-44">
                        <div className="space-y-2">
                          {[
                            { id: '001', symbol: 'AAPL', side: 'BUY', qty: 100, price: 150.25, status: 'Working' },
                            { id: '002', symbol: 'MSFT', side: 'SELL', qty: 50, price: 335.50, status: 'Filled' },
                            { id: '003', symbol: 'GOOGL', side: 'BUY', qty: 25, price: 2750.00, status: 'Partial' }
                          ].map((order) => (
                            <div key={order.id} className="text-xs border border-gray-700 rounded p-2">
                              <div className="flex justify-between">
                                <span className="text-cyan-400">{order.symbol}</span>
                                <Badge 
                                  variant="outline" 
                                  className={`text-xs ${
                                    order.status === 'Filled' ? 'border-green-400 text-green-400' :
                                    order.status === 'Working' ? 'border-yellow-400 text-yellow-400' :
                                    'border-blue-400 text-blue-400'
                                  }`}
                                >
                                  {order.status}
                                </Badge>
                              </div>
                              <div className="flex justify-between text-gray-400">
                                <span>{order.side} {order.qty}</span>
                                <span>@{order.price}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </TabsContent>
                    
                    <TabsContent value="fills" className="px-2">
                      <ScrollArea className="h-44">
                        <div className="text-xs text-gray-400 text-center py-8">
                          Recent executions will appear here
                        </div>
                      </ScrollArea>
                    </TabsContent>

                    <TabsContent value="blotter" className="px-2">
                      <ScrollArea className="h-44">
                        <div className="text-xs text-gray-400 text-center py-8">
                          Trading blotter and audit trail
                        </div>
                      </ScrollArea>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>

            {/* News and Events */}
            <div className="flex-1 p-4">
              <Card className="h-full bg-gray-800 border-gray-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-orange-400 flex items-center">
                    <Globe className="w-4 h-4 mr-2" />
                    Market News
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-56">
                    {[
                      { 
                        time: '14:23', 
                        headline: 'Federal Reserve signals potential rate cut in December meeting',
                        source: 'Reuters',
                        impact: 'high'
                      },
                      { 
                        time: '13:45', 
                        headline: 'Tech sector earnings beat expectations with AI growth',
                        source: 'Bloomberg',
                        impact: 'medium'
                      },
                      { 
                        time: '13:12', 
                        headline: 'Oil prices surge 3% on Middle East supply concerns',
                        source: 'WSJ',
                        impact: 'medium'
                      },
                      { 
                        time: '12:58', 
                        headline: 'Consumer confidence index reaches 18-month high',
                        source: 'MarketWatch',
                        impact: 'low'
                      },
                      { 
                        time: '12:34', 
                        headline: 'Cryptocurrency market cap surpasses $2.5 trillion',
                        source: 'CoinDesk',
                        impact: 'low'
                      }
                    ].map((news, index) => (
                      <div key={index} className="p-3 border-b border-gray-700 hover:bg-gray-700 cursor-pointer">
                        <div className="flex justify-between items-start mb-1">
                          <span className="text-xs text-gray-400">{news.time}</span>
                          <Badge 
                            variant="outline" 
                            className={`text-xs ${
                              news.impact === 'high' ? 'border-red-400 text-red-400' :
                              news.impact === 'medium' ? 'border-yellow-400 text-yellow-400' :
                              'border-gray-400 text-gray-400'
                            }`}
                          >
                            {news.impact.toUpperCase()}
                          </Badge>
                        </div>
                        <div className="text-xs text-gray-300 mb-1 leading-relaxed">
                          {news.headline}
                        </div>
                        <div className="text-xs text-gray-500">{news.source}</div>
                      </div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className={`h-6 ${terminalTheme.panel} border-t border-gray-700 flex items-center justify-between px-4 text-xs`}>
        <div className="flex items-center space-x-4">
          <span className="text-green-400">● Market Data Connected</span>
          <span className="text-gray-400">Last Update: {currentTime.toLocaleTimeString()}</span>
          <span className="text-gray-400">Orders: {positions.length} Active</span>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-gray-400">CPU: {systemMetrics.cpuUsage}%</span>
          <span className="text-gray-400">Memory: {systemMetrics.memoryUsage}%</span>
          <span className="text-gray-400">Latency: {systemMetrics.latency}ms</span>
        </div>
      </div>
    </div>
  );
}; 