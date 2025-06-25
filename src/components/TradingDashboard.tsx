
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
import { PortfolioAnalytics } from './PortfolioAnalytics';
import { QuantResearch } from './QuantResearch';
import { DerivativesTrading } from './DerivativesTrading';
import { InstitutionalOMS } from './InstitutionalOMS';
import { AlternativeData } from './AlternativeData';
import { RealTimeDataPipeline } from './RealTimeDataPipeline';
import { MLModelPipeline } from './MLModelPipeline';
import { AdvancedCharting } from './AdvancedCharting';
import { MarketMicrostructure } from './MarketMicrostructure';
import { HeaderBar } from './HeaderBar';
import { CommandBar } from './CommandBar';
import { StatusBar } from './StatusBar';
import { FloatingActionPanel } from './FloatingActionPanel';

export const TradingDashboard = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [layout, setLayout] = useState('professional');
  const [theme, setTheme] = useState('dark');
  const [notifications, setNotifications] = useState(3);
  const [commandInput, setCommandInput] = useState('');

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-terminal-bg via-terminal-bg to-terminal-panel text-terminal-text font-mono">
      {/* Enhanced Bloomberg-style Header */}
      <div className="h-16 bg-gradient-to-r from-terminal-panel via-terminal-panel to-terminal-bg border-b-2 border-terminal-orange flex flex-col shadow-lg backdrop-blur-sm">
        <HeaderBar 
          currentTime={currentTime}
          notifications={notifications}
          layout={layout}
          setLayout={setLayout}
        />
        <CommandBar 
          commandInput={commandInput}
          setCommandInput={setCommandInput}
          onCommandSubmit={handleCommandSubmit}
          setShowCommandPalette={setShowCommandPalette}
        />
      </div>

      {/* Enhanced Main Dashboard Grid - Now with Advanced Infrastructure */}
      <div className="grid grid-cols-24 gap-1 p-1 h-[calc(100vh-64px)] overflow-hidden">
        {/* Market Overview - Top Row */}
        <div className="col-span-24 h-20">
          <MarketOverview />
        </div>

        {/* Left Column - Data Pipeline & Infrastructure */}
        <div className="col-span-4 space-y-1 overflow-hidden">
          <div className="h-[40%]">
            <RealTimeDataPipeline />
          </div>
          <div className="h-[35%]">
            <Watchlist 
              selectedSymbol={selectedSymbol}
              onSymbolSelect={setSelectedSymbol}
            />
          </div>
          <div className="h-[23%]">
            <MarketMicrostructure />
          </div>
        </div>

        {/* Center-Left Column - Advanced Charting */}
        <div className="col-span-6 space-y-1">
          <div className="h-[70%]">
            <AdvancedCharting />
          </div>
          <div className="h-[28%]">
            <TechnicalIndicators symbol={selectedSymbol} />
          </div>
        </div>

        {/* Center Column - AI/ML Pipeline */}
        <div className="col-span-5 space-y-1">
          <div className="h-[45%]">
            <MLModelPipeline />
          </div>
          <div className="h-[25%]">
            <AIMarketAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[28%]">
            <AlternativeData />
          </div>
        </div>

        {/* Center-Right Column - Quant Research & Analytics */}
        <div className="col-span-5 space-y-1">
          <div className="h-[40%]">
            <QuantResearch />
          </div>
          <div className="h-[30%]">
            <PortfolioAnalytics />
          </div>
          <div className="h-[28%]">
            <DerivativesTrading />
          </div>
        </div>

        {/* Right Column - Institutional Trading & Risk */}
        <div className="col-span-4 space-y-1">
          <div className="h-[25%]">
            <InstitutionalOMS />
          </div>
          <div className="h-[20%]">
            <TradingInterface symbol={selectedSymbol} />
          </div>
          <div className="h-[20%]">
            <RiskManager />
          </div>
          <div className="h-[15%]">
            <SentimentAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[18%]">
            <OptionsFlow symbol={selectedSymbol} />
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
      <StatusBar currentTime={currentTime} />

      {/* Enhanced Floating Action Panel */}
      <FloatingActionPanel />
    </div>
  );
};
