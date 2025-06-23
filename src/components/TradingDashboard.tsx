
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
  PieChart, Activity, Target, Layers, Monitor
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";

export const TradingDashboard = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [layout, setLayout] = useState('professional');
  const [theme, setTheme] = useState('dark');
  const [notifications, setNotifications] = useState(3);

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
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const layouts = [
    { id: 'professional', name: 'Professional', icon: Layout },
    { id: 'analytical', name: 'Analytical', icon: BarChart },
    { id: 'compact', name: 'Compact', icon: Monitor },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-terminal-bg via-terminal-bg to-terminal-panel text-terminal-text">
      {/* Enhanced Header */}
      <div className="h-14 bg-gradient-to-r from-terminal-panel via-terminal-panel to-terminal-bg border-b border-terminal-border flex items-center justify-between px-6 shadow-lg backdrop-blur-sm">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Terminal className="w-6 h-6 text-terminal-orange animate-pulse-subtle" />
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
            </div>
            <div>
              <span className="font-bold text-xl text-terminal-orange bg-gradient-to-r from-terminal-orange to-terminal-amber bg-clip-text text-transparent">
                QuantTerminal
              </span>
              <div className="text-xs text-terminal-muted">Professional Edition</div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
              <span className="text-terminal-green font-medium">LIVE</span>
            </div>
            <div className="text-terminal-muted">
              {currentTime.toLocaleTimeString()} EST
            </div>
            <div className="text-terminal-cyan">
              SPX: 4,521.23 <span className="text-terminal-green">+12.45</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
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

          {/* Search */}
          <button 
            onClick={() => setShowCommandPalette(true)}
            className="terminal-button flex items-center space-x-2 hover:glow-cyan"
          >
            <Search className="w-4 h-4" />
            <span className="hidden md:inline">Search</span>
            <kbd className="hidden md:inline bg-terminal-border px-2 py-1 text-xs rounded">⌘K</kbd>
          </button>

          {/* Notifications */}
          <button className="terminal-button relative hover:glow-orange">
            <Bell className="w-4 h-4" />
            {notifications > 0 && (
              <span className="absolute -top-1 -right-1 bg-terminal-red text-xs rounded-full w-5 h-5 flex items-center justify-center text-white font-bold">
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
              <DropdownMenuLabel>Settings</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="hover:bg-terminal-border">
                <Shield className="w-4 h-4 mr-2" />
                Security
              </DropdownMenuItem>
              <DropdownMenuItem className="hover:bg-terminal-border">
                <Monitor className="w-4 h-4 mr-2" />
                Display
              </DropdownMenuItem>
              <DropdownMenuItem className="hover:bg-terminal-border">
                <Zap className="w-4 h-4 mr-2" />
                Performance
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Main Dashboard Grid - Professional Layout */}
      <div className="grid grid-cols-16 gap-1 p-1 h-[calc(100vh-56px)] overflow-hidden">
        {/* Market Overview - Top Row */}
        <div className="col-span-16 h-20">
          <MarketOverview />
        </div>

        {/* Left Column - Watchlist & Portfolio */}
        <div className="col-span-3 space-y-1 overflow-hidden">
          <div className="h-[45%]">
            <Watchlist 
              selectedSymbol={selectedSymbol}
              onSymbolSelect={setSelectedSymbol}
            />
          </div>
          <div className="h-[30%]">
            <PortfolioSummary />
          </div>
          <div className="h-[23%]">
            <RiskManager />
          </div>
        </div>

        {/* Center-Left Column - Charts */}
        <div className="col-span-6 space-y-1">
          <div className="h-[60%]">
            <PriceChart symbol={selectedSymbol} />
          </div>
          <div className="h-[38%]">
            <TechnicalIndicators symbol={selectedSymbol} />
          </div>
        </div>

        {/* Center-Right Column - AI & Analysis */}
        <div className="col-span-4 space-y-1">
          <div className="h-[35%]">
            <AIMarketAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[30%]">
            <SentimentAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[33%]">
            <OptionsFlow symbol={selectedSymbol} />
          </div>
        </div>

        {/* Right Column - Trading & News */}
        <div className="col-span-3 space-y-1">
          <div className="h-[40%]">
            <TradingInterface symbol={selectedSymbol} />
          </div>
          <div className="h-[30%]">
            <OrderBook symbol={selectedSymbol} />
          </div>
          <div className="h-[28%]">
            <NewsFeed />
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

      {/* Floating Action Panel */}
      <div className="fixed bottom-4 right-4 flex flex-col space-y-2">
        <button className="bg-terminal-orange hover:bg-terminal-amber text-terminal-bg p-3 rounded-full shadow-lg glow-orange transition-all duration-300 hover:scale-110">
          <Brain className="w-5 h-5" />
        </button>
        <button className="bg-terminal-cyan hover:bg-terminal-cyan/80 text-terminal-bg p-3 rounded-full shadow-lg glow-cyan transition-all duration-300 hover:scale-110">
          <Target className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};
