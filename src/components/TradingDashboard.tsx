import React, { useState, useEffect } from 'react';
import { MarketOverview } from './MarketOverview';
import { PriceChart } from './PriceChart';
import { Watchlist } from './Watchlist';
import { NewsFeed } from './NewsFeed';
import { OrderBook } from './OrderBook';
import { PortfolioSummary } from './PortfolioSummary';
import { CommandPalette } from './CommandPalette';
import { TechnicalIndicators } from './TechnicalIndicators';
import { AIMarketAnalysis } from './AIMarketAnalysis';
import { SentimentAnalysis } from './SentimentAnalysis';
import { OptionsFlow } from './OptionsFlow';
import { TradingInterface } from './TradingInterface';
import { AdvancedScreener } from './AdvancedScreener';
import { RiskManager } from './RiskManager';
import { 
  Search, Terminal, BarChart, Wallet, Bell, Settings, 
  Layout, Brain, TrendingUp, Globe, Shield, Zap,
  PieChart, Activity, Target, Layers, Monitor, Command
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ProfessionalTerminal } from './ProfessionalTerminal';
import { InstitutionalTradingTerminal } from './InstitutionalTradingTerminal';
import { QuantitativeAnalysis } from './QuantitativeAnalysis';
import { NewsSentimentFeed } from './NewsSentimentFeed';
import { GreeksDashboard } from './GreeksDashboard';
import { BloombergTerminalPro } from './BloombergTerminalPro';

export const TradingDashboard = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [layout, setLayout] = useState('professional');
  const [theme, setTheme] = useState('dark');
  const [notifications, setNotifications] = useState(3);
  const [commandInput, setCommandInput] = useState('');
  const [activeTab, setActiveTab] = useState('terminal');

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(true);
      }
      // Bloomberg-style function keys
      if (e.key === 'F8') {
        e.preventDefault();
        setSelectedSymbol('AAPL');
        console.log('F8: Equity mode activated');
      }
      if (e.key === 'F9') {
        e.preventDefault();
        console.log('F9: Government bonds activated');
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleCommandSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (commandInput.trim()) {
      // Parse Bloomberg-style commands
      const cmd = commandInput.toUpperCase();
      if (cmd.includes('EQUITY') || cmd.includes('<EQUITY>')) {
        const symbol = cmd.split(' ')[0];
        setSelectedSymbol(symbol);
      }
      console.log('Command executed:', commandInput);
      setCommandInput('');
    }
  };

  const layouts = [
    { id: 'professional', name: 'Professional', icon: Layout },
    { id: 'analytical', name: 'Analytical', icon: BarChart },
    { id: 'compact', name: 'Compact', icon: Monitor },
  ];

  const dashboardTabs = [
    {
      id: 'bloomberg',
      label: 'Bloomberg Terminal Pro',
      icon: <Terminal className="h-4 w-4" />,
      component: <BloombergTerminalPro />
    },
    {
      id: 'terminal',
      label: 'Professional Terminal',
      icon: <BarChart className="h-4 w-4" />,
      component: <ProfessionalTerminal />
    },
    {
      id: 'institutional',
      label: 'Institutional Trading',
      icon: <Zap className="h-4 w-4" />,
      component: <InstitutionalTradingTerminal />
    },
    {
      id: 'quantitative',
      label: 'Quantitative Analysis',
      icon: <BarChart className="h-4 w-4" />,
      component: <QuantitativeAnalysis symbol={selectedSymbol} />
    },
    {
      id: 'screener',
      label: 'Advanced Screener',
      icon: <Search className="h-4 w-4" />,
      component: <AdvancedScreener />
    },
    {
      id: 'ai-analysis',
      label: 'AI Market Analysis',
      icon: <Brain className="h-4 w-4" />,
      component: <AIMarketAnalysis symbol={selectedSymbol} />
    },
    {
      id: 'options',
      label: 'Options Flow',
      icon: <Target className="h-4 w-4" />,
      component: <OptionsFlow symbol={selectedSymbol} />
    },
    {
      id: 'greeks',
      label: 'Greeks Dashboard',
      icon: <Activity className="h-4 w-4" />,
      component: <GreeksDashboard />
    },
    {
      id: 'technical',
      label: 'Technical Analysis',
      icon: <TrendingUp className="h-4 w-4" />,
      component: <TechnicalIndicators symbol={selectedSymbol} />
    },
    {
      id: 'risk',
      label: 'Risk Management',
      icon: <Shield className="h-4 w-4" />,
      component: <RiskManager />
    },
    {
      id: 'news',
      label: 'News & Sentiment',
      icon: <Activity className="h-4 w-4" />,
      component: <NewsSentimentFeed />
    }
  ];

  const quickStats = [
    { label: 'Portfolio Value', value: '$1,250,000', change: '+2.45%', positive: true },
    { label: 'Day P&L', value: '+$12,450', change: '+0.99%', positive: true },
    { label: 'Open Positions', value: '23', change: '+3', positive: true },
    { label: 'Buying Power', value: '$485,000', change: '-$15K', positive: false }
  ];

  const marketOverview = [
    { symbol: 'SPY', price: 445.67, change: 2.34, changePercent: 0.53 },
    { symbol: 'QQQ', price: 378.90, change: -1.23, changePercent: -0.32 },
    { symbol: 'IWM', price: 198.45, change: 0.89, changePercent: 0.45 },
    { symbol: 'VIX', price: 18.67, change: -0.45, changePercent: -2.35 }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-gray-900">Quant Bloom Nexus</h1>
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              LIVE MARKET
            </Badge>
          </div>
          
          <div className="flex items-center space-x-6">
            {/* Quick Stats */}
            <div className="flex items-center space-x-6">
              {quickStats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-xs text-gray-500">{stat.label}</div>
                  <div className="font-semibold text-gray-900">{stat.value}</div>
                  <div className={`text-xs ${stat.positive ? 'text-green-600' : 'text-red-600'}`}>
                    {stat.change}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="border-l border-gray-200 pl-6">
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>

        {/* Market Overview */}
        <div className="mt-4 flex items-center space-x-8">
          <span className="text-sm font-medium text-gray-700">Market Overview:</span>
          {marketOverview.map((market, index) => (
            <div key={index} className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-900">{market.symbol}</span>
              <span className="text-sm text-gray-600">${market.price.toFixed(2)}</span>
              <span className={`text-sm ${market.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {market.change >= 0 ? '+' : ''}{market.change.toFixed(2)} ({market.changePercent.toFixed(2)}%)
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Symbol Selector */}
      <div className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700">Active Symbol:</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM'].map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          
          <div className="ml-8 flex items-center space-x-2">
            <span className="text-sm text-gray-500">Last Update:</span>
            <span className="text-sm font-medium text-gray-900">
              {new Date().toLocaleTimeString()}
            </span>
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Main Dashboard */}
      <div className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-5 lg:grid-cols-10 w-full mb-6">
            {dashboardTabs.map((tab) => (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                className="flex items-center space-x-2 text-xs"
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {dashboardTabs.map((tab) => (
            <TabsContent key={tab.id} value={tab.id} className="mt-0">
              <Card className="border-0 shadow-lg">
                <CardContent className="p-0">
                  {tab.component}
                </CardContent>
              </Card>
            </TabsContent>
          ))}
        </Tabs>
      </div>

      {/* Footer */}
      <div className="bg-white border-t border-gray-200 px-6 py-3">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-6">
            <span>© 2024 Quant Bloom Nexus</span>
            <span>Market Data: Real-time</span>
            <span>AI Models: Active</span>
            <span>Risk Engine: Operational</span>
          </div>
          <div className="flex items-center space-x-4">
            <span>Latency: 12ms</span>
            <span>CPU: 23%</span>
            <span>Memory: 45%</span>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>All Systems Operational</span>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Bloomberg-style Header */}
      <div className="h-16 bg-gradient-to-r from-terminal-panel via-terminal-panel to-terminal-bg border-b-2 border-terminal-orange flex flex-col shadow-lg backdrop-blur-sm">
        {/* Top Header Bar */}
        <div className="h-10 flex items-center justify-between px-6 border-b border-terminal-border/50">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Terminal className="w-6 h-6 text-terminal-orange animate-pulse-subtle" />
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
              </div>
              <div>
                <span className="font-bold text-xl text-terminal-orange bg-gradient-to-r from-terminal-orange to-terminal-amber bg-clip-text text-transparent">
                  BLOOMBERG TERMINAL
                </span>
                <div className="text-xs text-terminal-muted">Professional Edition v12.8.4</div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
                <span className="text-terminal-green font-medium">LIVE MARKET DATA</span>
              </div>
              <div className="text-terminal-muted">
                {currentTime.toLocaleTimeString()} EST
              </div>
              <div className="text-terminal-cyan font-mono">
                SPX: 4,521.23 <span className="text-terminal-green">+12.45</span>
              </div>
              <div className="text-terminal-cyan font-mono">
                NDX: 15,245.67 <span className="text-terminal-red">-8.23</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Function Key Shortcuts Display */}
            <div className="flex items-center space-x-2 text-xs text-terminal-muted">
              <kbd className="bg-terminal-border px-2 py-1 rounded">F8</kbd>
              <span>Equity</span>
              <kbd className="bg-terminal-border px-2 py-1 rounded">F9</kbd>
              <span>Bonds</span>
            </div>

            {/* Layout Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger className="terminal-button flex items-center space-x-2 hover:glow-orange">
                <Layout className="w-4 h-4" />
                <span className="hidden md:inline">Layout</span>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
                <DropdownMenuLabel>Choose Layout</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {layouts.map((layoutOption) => (
                  <DropdownMenuItem 
                    key={layoutOption.id}
                    onClick={() => setLayout(layoutOption.id)}
                    className="hover:bg-terminal-border"
                  >
                    <layoutOption.icon className="w-4 h-4 mr-2" />
                    {layoutOption.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Notifications */}
            <button className="terminal-button relative hover:glow-orange">
              <Bell className="w-4 h-4" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 bg-terminal-red text-xs rounded-full w-5 h-5 flex items-center justify-center text-white font-bold animate-pulse">
                  {notifications}
                </span>
              )}
            </button>

            {/* Settings */}
            <DropdownMenu>
              <DropdownMenuTrigger className="terminal-button hover:glow-cyan">
                <Settings className="w-4 h-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
                <DropdownMenuLabel>Terminal Settings</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="hover:bg-terminal-border">
                  <Shield className="w-4 h-4 mr-2" />
                  Security Settings
                </DropdownMenuItem>
                <DropdownMenuItem className="hover:bg-terminal-border">
                  <Monitor className="w-4 h-4 mr-2" />
                  Display Preferences
                </DropdownMenuItem>
                <DropdownMenuItem className="hover:bg-terminal-border">
                  <Zap className="w-4 h-4 mr-2" />
                  Performance Tuning
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Bloomberg-style Command Line */}
        <div className="h-6 flex items-center px-6 bg-gradient-to-r from-terminal-bg to-terminal-panel border-b border-terminal-border/30">
          <div className="flex items-center space-x-2 w-full">
            <Command className="w-3 h-3 text-terminal-orange" />
            <form onSubmit={handleCommandSubmit} className="flex-1">
              <input
                type="text"
                value={commandInput}
                onChange={(e) => setCommandInput(e.target.value)}
                placeholder="Enter command (e.g., AAPL <Equity> GP for price graph)..."
                className="bg-transparent text-terminal-text placeholder-terminal-muted text-xs font-mono w-full focus:outline-none border-none"
              />
            </form>
            <button 
              onClick={() => setShowCommandPalette(true)}
              className="flex items-center space-x-1 text-xs text-terminal-muted hover:text-terminal-orange transition-colors"
            >
              <Search className="w-3 h-3" />
              <kbd className="bg-terminal-border px-1 rounded text-xs">⌘K</kbd>
            </button>
          </div>
        </div>
      </div>

      {/* Command Palette */}
      {showCommandPalette && (
        <CommandPalette 
          onClose={() => setShowCommandPalette(false)}
          onSymbolSelect={setSelectedSymbol}
        />
      )}

      {/* Bloomberg-style Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 h-6 bg-terminal-panel border-t border-terminal-border flex items-center justify-between px-4 text-xs text-terminal-muted">
        <div className="flex items-center space-x-4">
          <span>Connected to Market Data Feed</span>
          <span className="text-terminal-green">●</span>
          <span>Last Update: {currentTime.toLocaleTimeString()}</span>
        </div>
        <div className="flex items-center space-x-4">
          <span>CPU: 12%</span>
          <span>Memory: 2.1GB</span>
          <span>Latency: 0.8ms</span>
        </div>
      </div>

      {/* Enhanced Floating Action Panel */}
      <div className="fixed bottom-8 right-4 flex flex-col space-y-2">
        <button className="bg-terminal-orange hover:bg-terminal-amber text-terminal-bg p-3 rounded-full shadow-2xl glow-orange transition-all duration-300 hover:scale-110 group">
          <Brain className="w-5 h-5" />
          <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            AI Analysis
          </div>
        </button>
        <button className="bg-terminal-cyan hover:bg-terminal-cyan/80 text-terminal-bg p-3 rounded-full shadow-2xl glow-cyan transition-all duration-300 hover:scale-110 group">
          <Target className="w-5 h-5" />
          <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            Quick Trade
          </div>
        </button>
        <button className="bg-terminal-green hover:bg-terminal-green/80 text-terminal-bg p-3 rounded-full shadow-2xl transition-all duration-300 hover:scale-110 group">
          <Activity className="w-5 h-5" />
          <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            Live Activity
          </div>
        </button>
      </div>
    </div>
  );
};
