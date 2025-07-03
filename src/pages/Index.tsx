import { useState } from "react";
import { ProfessionalTerminal } from "@/components/ProfessionalTerminal";
import { BloombergTerminalPro } from "@/components/BloombergTerminalPro";
import { InstitutionalTradingTerminal } from "@/components/InstitutionalTradingTerminal";
import { QuantitativeAnalysis } from "@/components/QuantitativeAnalysis";
import { AdvancedScreener } from "@/components/AdvancedScreener";
import { AIMarketAnalysis } from "@/components/AIMarketAnalysis";
import { OptionsFlow } from "@/components/OptionsFlow";
import { GreeksDashboard } from "@/components/GreeksDashboard";
import { RiskManager } from "@/components/RiskManager";
import { NewsFeed } from "@/components/NewsFeed";
import { NewsSentimentFeed } from "@/components/NewsSentimentFeed";
import { TradingDashboard } from "@/components/TradingDashboard";
import { TradingInterface } from "@/components/TradingInterface";
import { PortfolioSummary } from "@/components/PortfolioSummary";
import { MarketOverview } from "@/components/MarketOverview";
import { Watchlist } from "@/components/Watchlist";
import { SentimentAnalysis } from "@/components/SentimentAnalysis";
import { DashboardProvider } from "@/context/DashboardContext";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type ActiveView = 
  | 'bloomberg' 
  | 'professional' 
  | 'institutional' 
  | 'quantitative' 
  | 'screener' 
  | 'ai-analysis' 
  | 'options' 
  | 'greeks' 
  | 'risk' 
  | 'news' 
  | 'sentiment' 
  | 'trading' 
  | 'portfolio' 
  | 'market' 
  | 'watchlist';

const Index = () => {
  const [activeView, setActiveView] = useState<ActiveView>('bloomberg');

  const navigationItems = [
    { id: 'bloomberg', label: 'Bloomberg Terminal Pro', icon: '🏛️' },
    { id: 'professional', label: 'Professional Terminal', icon: '💼' },
    { id: 'institutional', label: 'Institutional Trading', icon: '🏢' },
    { id: 'quantitative', label: 'Quantitative Analysis', icon: '📊' },
    { id: 'screener', label: 'Advanced Screener', icon: '🔍' },
    { id: 'ai-analysis', label: 'AI Market Analysis', icon: '🤖' },
    { id: 'options', label: 'Options Flow', icon: '📈' },
    { id: 'greeks', label: 'Greeks Dashboard', icon: '🔢' },
    { id: 'risk', label: 'Risk Manager', icon: '⚠️' },
    { id: 'news', label: 'News Feed', icon: '📰' },
    { id: 'sentiment', label: 'Sentiment Analysis', icon: '😊' },
    { id: 'trading', label: 'Trading Dashboard', icon: '💹' },
    { id: 'portfolio', label: 'Portfolio Summary', icon: '💼' },
    { id: 'market', label: 'Market Overview', icon: '🌐' },
    { id: 'watchlist', label: 'Watchlist', icon: '👁️' }
  ];

  const renderActiveComponent = () => {
    switch (activeView) {
      case 'bloomberg':
        return <BloombergTerminalPro />;
      case 'professional':
        return <ProfessionalTerminal />;
      case 'institutional':
        return <InstitutionalTradingTerminal />;
      case 'quantitative':
        return <QuantitativeAnalysis symbol="AAPL" />;
      case 'screener':
        return <AdvancedScreener />;
      case 'ai-analysis':
        return <AIMarketAnalysis symbol="AAPL" />;
      case 'options':
        return <OptionsFlow symbol="AAPL" />;
      case 'greeks':
        return <GreeksDashboard />;
      case 'risk':
        return <RiskManager />;
      case 'news':
        return <NewsFeed />;
      case 'sentiment':
        return <SentimentAnalysis symbol="AAPL" />;
      case 'trading':
        return <TradingDashboard />;
      case 'portfolio':
        return <PortfolioSummary />;
      case 'market':
        return <MarketOverview />;
      case 'watchlist':
        return <Watchlist selectedSymbol="AAPL" onSymbolSelect={(symbol) => console.log('Selected:', symbol)} />;
      default:
        return <BloombergTerminalPro />;
    }
  };

  return (
    <DashboardProvider>
      <div className="w-full flex flex-col h-screen bg-black text-orange-400">
        {/* Main Navigation Header */}
        <div className="bg-gray-900 border-b-2 border-orange-500 p-2">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-xl font-bold text-orange-400">QUANT BLOOM NEXUS - PROFESSIONAL TRADING PLATFORM</h1>
            <div className="text-sm text-gray-400">
              {new Date().toLocaleString()} EST | All Systems Operational
            </div>
          </div>
          
          {/* Navigation Tabs */}
          <div className="flex flex-wrap gap-1">
            {navigationItems.map((item) => (
              <button
                key={item.id}
                className={`px-3 py-1 text-xs font-bold rounded transition-colors ${
                  activeView === item.id 
                    ? 'bg-orange-500 text-black' 
                    : 'bg-gray-800 text-orange-400 hover:bg-gray-700'
                }`}
                onClick={() => setActiveView(item.id as ActiveView)}
              >
                {item.icon} {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-hidden">
          {renderActiveComponent()}
        </div>

        {/* Status Bar */}
        <div className="bg-gray-900 border-t border-orange-500 px-4 py-1 text-xs">
          <div className="flex justify-between items-center">
            <div className="flex space-x-4">
              <span>Active Module: {navigationItems.find(item => item.id === activeView)?.label}</span>
              <span>Backend: Connected</span>
              <span>AI Models: Operational</span>
              <span>Market Data: Live</span>
            </div>
            <div className="flex space-x-4">
              <span>© 2024 Quant Bloom Nexus</span>
              <span>Professional Trading Terminal</span>
            </div>
          </div>
        </div>
      </div>
    </DashboardProvider>
  );
};

export default Index;
