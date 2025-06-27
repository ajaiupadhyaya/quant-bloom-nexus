
import React, { useState, useEffect, useCallback } from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// Import your components
import { AdvancedPriceChart } from './AdvancedPriceChart';
import { OrderBook } from './OrderBook';
import { NewsFeed } from './NewsFeed';
import { NewsSentimentFeed } from './NewsSentimentFeed';
import { Watchlist } from './Watchlist';
import { PortfolioSummary } from './PortfolioSummary';
import { TradingInterface } from './TradingInterface';
import { MarketOverview } from './MarketOverview';
import { TechnicalIndicators } from './TechnicalIndicators';
import { AIMarketAnalysis } from './AIMarketAnalysis';
import { SentimentAnalysis } from './SentimentAnalysis';
import { OptionsFlow } from './OptionsFlow';
import { AdvancedScreener } from './AdvancedScreener';
import { RiskManager } from './RiskManager';
import { GreeksDashboard } from './GreeksDashboard';

const ResponsiveGridLayout = WidthProvider(Responsive);

// Define a type for your layout items
interface LayoutItem {
  i: string; // ID of the item
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  maxW?: number;
  minH?: number;
  maxH?: number;
  static?: boolean;
}

// Define a type for your component map
interface ComponentMap {
  [key: string]: React.ComponentType<any>;
}

// Map component names to their actual components
const componentMap: ComponentMap = {
  PriceChart: AdvancedPriceChart,
  OrderBook: OrderBook,
  NewsFeed: NewsSentimentFeed,
  Watchlist: Watchlist,
  PortfolioSummary: PortfolioSummary,
  TradingInterface: TradingInterface,
  MarketOverview: MarketOverview,
  TechnicalIndicators: TechnicalIndicators,
  AIMarketAnalysis: AIMarketAnalysis,
  SentimentAnalysis: SentimentAnalysis,
  OptionsFlow: OptionsFlow,
  AdvancedScreener: AdvancedScreener,
  RiskManager: RiskManager,
  GreeksDashboard: GreeksDashboard,
};

// Initial layout for different breakpoints
const initialLayouts: { [key: string]: LayoutItem[] } = {
  lg: [
    { i: 'market-overview', x: 0, y: 0, w: 12, h: 1, minH: 1, maxH: 1 },
    { i: 'watchlist', x: 0, y: 1, w: 3, h: 6 },
    { i: 'price-chart', x: 3, y: 1, w: 6, h: 6 },
    { i: 'order-book', x: 9, y: 1, w: 3, h: 3 },
    { i: 'news-sentiment-feed', x: 9, y: 4, w: 3, h: 3 },
    { i: 'portfolio-summary', x: 0, y: 7, w: 3, h: 3 },
    { i: 'trading-interface', x: 3, y: 7, w: 6, h: 3 },
    { i: 'technical-indicators', x: 9, y: 7, w: 3, h: 3 },
    { i: 'greeks-dashboard', x: 0, y: 10, w: 3, h: 3 },
  ],
  md: [
    { i: 'market-overview', x: 0, y: 0, w: 10, h: 1, minH: 1, maxH: 1 },
    { i: 'watchlist', x: 0, y: 1, w: 5, h: 5 },
    { i: 'price-chart', x: 5, y: 1, w: 5, h: 5 },
    { i: 'order-book', x: 0, y: 6, w: 5, h: 3 },
    { i: 'news-feed', x: 5, y: 6, w: 5, h: 3 },
  ],
  sm: [
    { i: 'market-overview', x: 0, y: 0, w: 6, h: 1, minH: 1, maxH: 1 },
    { i: 'watchlist', x: 0, y: 1, w: 6, h: 4 },
    { i: 'price-chart', x: 0, y: 5, w: 6, h: 4 },
  ],
};

interface DashboardGridProps {
  selectedSymbol: string;
  onSymbolSelect: (symbol: string) => void;
}

export const DashboardGrid: React.FC<DashboardGridProps> = ({ selectedSymbol, onSymbolSelect }) => {
  const [layouts, setLayouts] = useState<{ [key: string]: LayoutItem[] }>(() => {
    // Load layout from local storage or use initial layout
    const savedLayout = localStorage.getItem('dashboardLayout');
    return savedLayout ? JSON.parse(savedLayout) : initialLayouts;
  });

  const onLayoutChange = useCallback((currentLayout: LayoutItem[], allLayouts: { [key: string]: LayoutItem[] }) => {
    setLayouts(allLayouts);
  }, []);

  useEffect(() => {
    // Save layout to local storage whenever it changes
    localStorage.setItem('dashboardLayout', JSON.stringify(layouts));
  }, [layouts]);

  const generateLayout = (breakpoint: string) => {
    return layouts[breakpoint] || initialLayouts[breakpoint];
  };

  const getComponentProps = (componentId: string) => {
    // Pass necessary props to the components
    switch (componentId) {
      case 'price-chart':
      case 'technical-indicators':
      case 'ai-market-analysis':
      case 'sentiment-analysis':
      case 'options-flow':
      case 'trading-interface':
      case 'order-book':
        return { symbol: selectedSymbol };
      case 'watchlist':
        return { selectedSymbol, onSymbolSelect };
      default:
        return {};
    }
  };

  return (
    <ResponsiveGridLayout
      className="layout"
      layouts={layouts}
      onLayoutChange={onLayoutChange}
      breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
      cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
      rowHeight={100} // Adjust row height as needed
      margin={[10, 10]} // Margin between grid items
      containerPadding={[10, 10]} // Padding around the grid container
      useCSSTransforms={true} // Use CSS transforms for better performance
    >
      {Object.keys(componentMap).map((componentName) => {
        const Component = componentMap[componentName];
        const layoutItem = layouts.lg.find(item => item.i === componentName.toLowerCase().replace(/([A-Z])/g, '-$1').toLowerCase());

        if (!layoutItem) return null; // Only render components that are in the initial layout

        return (
          <div key={layoutItem.i} data-grid={layoutItem} className="terminal-panel overflow-hidden">
            <Component {...getComponentProps(layoutItem.i)} />
          </div>
        );
      })}
    </ResponsiveGridLayout>
  );
};
