
import React, { useState, useEffect } from 'react';
import { MarketOverview } from './MarketOverview';
import { PriceChart } from './PriceChart';
import { Watchlist } from './Watchlist';
import { NewsFeed } from './NewsFeed';
import { OrderBook } from './OrderBook';
import { PortfolioSummary } from './PortfolioSummary';
import { CommandPalette } from './CommandPalette';
import { Search, Terminal, BarChart, Wallet, Bell, Settings } from 'lucide-react';

export const TradingDashboard = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

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

  return (
    <div className="min-h-screen bg-terminal-bg text-terminal-text">
      {/* Header */}
      <div className="h-12 bg-terminal-panel border-b border-terminal-border flex items-center justify-between px-4">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <Terminal className="w-5 h-5 text-terminal-orange" />
            <span className="font-bold text-terminal-orange">QuantTerminal</span>
          </div>
          <div className="text-sm text-terminal-muted">
            {currentTime.toLocaleTimeString()} EST
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button 
            onClick={() => setShowCommandPalette(true)}
            className="terminal-button flex items-center space-x-2"
          >
            <Search className="w-4 h-4" />
            <span className="hidden md:inline">Search</span>
            <kbd className="hidden md:inline bg-terminal-border px-1.5 py-0.5 text-xs rounded">⌘K</kbd>
          </button>
          <button className="terminal-button">
            <Bell className="w-4 h-4" />
          </button>
          <button className="terminal-button">
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-12 gap-1 p-1 h-[calc(100vh-48px)]">
        {/* Market Overview - Top Row */}
        <div className="col-span-12 h-20">
          <MarketOverview />
        </div>

        {/* Left Column */}
        <div className="col-span-3 space-y-1">
          <div className="h-80">
            <Watchlist 
              selectedSymbol={selectedSymbol}
              onSymbolSelect={setSelectedSymbol}
            />
          </div>
          <div className="h-60">
            <PortfolioSummary />
          </div>
        </div>

        {/* Center Column */}
        <div className="col-span-6 space-y-1">
          <div className="h-96">
            <PriceChart symbol={selectedSymbol} />
          </div>
          <div className="h-44">
            <OrderBook symbol={selectedSymbol} />
          </div>
        </div>

        {/* Right Column */}
        <div className="col-span-3">
          <NewsFeed />
        </div>
      </div>

      {/* Command Palette */}
      {showCommandPalette && (
        <CommandPalette 
          onClose={() => setShowCommandPalette(false)}
          onSymbolSelect={setSelectedSymbol}
        />
      )}
    </div>
  );
};
