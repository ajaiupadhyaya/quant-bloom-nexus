// @ts-ignore
import { create } from 'zustand';

export interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  side: 'long' | 'short';
}

export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  status: 'pending' | 'filled' | 'cancelled';
  timestamp: number;
}

export interface Portfolio {
  id: string;
  name: string;
  totalValue: number;
  cashBalance: number;
  dayPnL: number;
  totalPnL: number;
  positions: Position[];
}

export interface MarketDataPoint {
  symbol: string;
  timestamp: number;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  change: number;
  changePercent: number;
}

export interface TradingState {
  // Core State
  selectedSymbol: string;
  positions: Position[];
  orders: Order[];
  portfolios: Portfolio[];
  activePortfolio: string;
  marketData: Map<string, MarketDataPoint>;
  connectionStatus: 'connected' | 'disconnected';
  
  // UI State
  activeView: 'dashboard' | 'trading' | 'analysis' | 'research';
  layout: 'professional' | 'analytical' | 'compact';
  theme: 'dark' | 'light';
  notifications: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    title: string;
    message: string;
    timestamp: number;
    read: boolean;
  }>;

  // Actions
  setSelectedSymbol: (symbol: string) => void;
  addPosition: (position: Position) => void;
  updatePosition: (symbol: string, updates: Partial<Position>) => void;
  closePosition: (symbol: string) => void;
  createOrder: (order: Omit<Order, 'id' | 'timestamp'>) => void;
  updateOrder: (orderId: string, updates: Partial<Order>) => void;
  cancelOrder: (orderId: string) => void;
  updatePortfolio: (portfolioId: string, updates: Partial<Portfolio>) => void;
  setActivePortfolio: (portfolioId: string) => void;
  updateMarketData: (symbol: string, data: MarketDataPoint) => void;
  setConnectionStatus: (status: 'connected' | 'disconnected') => void;
  setActiveView: (view: TradingState['activeView']) => void;
  setLayout: (layout: TradingState['layout']) => void;
  setTheme: (theme: TradingState['theme']) => void;
  addNotification: (notification: Omit<TradingState['notifications'][0], 'id' | 'timestamp' | 'read'>) => void;
  markNotificationRead: (notificationId: string) => void;
  clearNotifications: () => void;
  calculatePortfolioValue: (portfolioId: string) => number;
  getTopMovers: (limit?: number) => MarketDataPoint[];
}

export const useTradingStore = create<TradingState>((set, get) => ({
  // Initial State
  selectedSymbol: 'AAPL',
  positions: [],
  orders: [],
  portfolios: [
    {
      id: 'default',
      name: 'Main Portfolio',
      totalValue: 100000,
      cashBalance: 50000,
      dayPnL: 0,
      totalPnL: 0,
      positions: []
    }
  ],
  activePortfolio: 'default',
  marketData: new Map(),
  connectionStatus: 'disconnected',
  
  activeView: 'dashboard',
  layout: 'professional',
  theme: 'dark',
  notifications: [],
  
  // Actions
  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
  
  addPosition: (position) => set((state) => {
    const existingIndex = state.positions.findIndex(p => p.symbol === position.symbol);
    if (existingIndex !== -1) {
      // Update existing position
      const existing = state.positions[existingIndex];
      const totalQuantity = existing.quantity + position.quantity;
      const avgPrice = ((existing.avgPrice * existing.quantity) + (position.avgPrice * position.quantity)) / totalQuantity;
      
      const updatedPositions = [...state.positions];
      updatedPositions[existingIndex] = {
        ...existing,
        quantity: totalQuantity,
        avgPrice: avgPrice
      };
      
      return { positions: updatedPositions };
    } else {
      return { positions: [...state.positions, position] };
    }
  }),
  
  updatePosition: (symbol, updates) => set((state) => {
    const positionIndex = state.positions.findIndex(p => p.symbol === symbol);
    if (positionIndex !== -1) {
      const updatedPositions = [...state.positions];
      updatedPositions[positionIndex] = { ...updatedPositions[positionIndex], ...updates };
      return { positions: updatedPositions };
    }
    return state;
  }),
  
  closePosition: (symbol) => set((state) => {
    const positionIndex = state.positions.findIndex(p => p.symbol === symbol);
    if (positionIndex !== -1) {
      const position = state.positions[positionIndex];
      const updatedPositions = state.positions.filter(p => p.symbol !== symbol);
      
      // Add to realized P&L
      const activePortfolio = state.portfolios.find(p => p.id === state.activePortfolio);
      if (activePortfolio) {
        const updatedPortfolios = state.portfolios.map(p => 
          p.id === state.activePortfolio 
            ? { 
                ...p, 
                totalPnL: p.totalPnL + position.unrealizedPnL,
                cashBalance: p.cashBalance + (position.quantity * position.currentPrice)
              }
            : p
        );
        return { positions: updatedPositions, portfolios: updatedPortfolios };
      }
      
      return { positions: updatedPositions };
    }
    return state;
  }),
  
  createOrder: (orderData) => set((state) => {
    const order: Order = {
      ...orderData,
      id: `order_${Date.now()}`,
      timestamp: Date.now(),
      status: 'pending'
    };
    
    const newNotification = {
      id: `order_${Date.now()}`,
      type: 'info' as const,
      title: 'Order Created',
      message: `${order.side.toUpperCase()} ${order.quantity} ${order.symbol}`,
      timestamp: Date.now(),
      read: false
    };
    
    return { 
      orders: [...state.orders, order],
      notifications: [newNotification, ...state.notifications]
    };
  }),
  
  updateOrder: (orderId, updates) => set((state) => {
    const orderIndex = state.orders.findIndex(o => o.id === orderId);
    if (orderIndex !== -1) {
      const updatedOrders = [...state.orders];
      updatedOrders[orderIndex] = { ...updatedOrders[orderIndex], ...updates };
      return { orders: updatedOrders };
    }
    return state;
  }),
  
  cancelOrder: (orderId) => set((state) => {
    return {
      orders: state.orders.map(order => 
        order.id === orderId ? { ...order, status: 'cancelled' as const } : order
      )
    };
  }),
  
  updatePortfolio: (portfolioId, updates) => set((state) => {
    return {
      portfolios: state.portfolios.map(portfolio => 
        portfolio.id === portfolioId ? { ...portfolio, ...updates } : portfolio
      )
    };
  }),
  
  setActivePortfolio: (portfolioId) => set({ activePortfolio: portfolioId }),
  
  updateMarketData: (symbol, data) => set((state) => {
    const newMarketData = new Map(state.marketData);
    newMarketData.set(symbol, data);
    
    // Update positions with current prices
    const updatedPositions = state.positions.map(position => {
      if (position.symbol === symbol) {
        const unrealizedPnL = (data.last - position.avgPrice) * position.quantity * (position.side === 'long' ? 1 : -1);
        return {
          ...position,
          currentPrice: data.last,
          unrealizedPnL
        };
      }
      return position;
    });
    
    return { 
      marketData: newMarketData,
      positions: updatedPositions
    };
  }),
  
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  
  setActiveView: (view) => set({ activeView: view }),
  
  setLayout: (layout) => set({ layout: layout }),
  
  setTheme: (theme) => set({ theme: theme }),
  
  addNotification: (notificationData) => set((state) => {
    const notification = {
      ...notificationData,
      id: `notification_${Date.now()}`,
      timestamp: Date.now(),
      read: false
    };
    return {
      notifications: [notification, ...state.notifications.slice(0, 99)]
    };
  }),
  
  markNotificationRead: (notificationId) => set((state) => {
    return {
      notifications: state.notifications.map(notification => 
        notification.id === notificationId ? { ...notification, read: true } : notification
      )
    };
  }),
  
  clearNotifications: () => set({ notifications: [] }),
  
  // Utility functions
  calculatePortfolioValue: (portfolioId) => {
    const state = get();
    const portfolio = state.portfolios.find(p => p.id === portfolioId);
    if (!portfolio) return 0;
    
    let totalValue = portfolio.cashBalance;
    
    state.positions.forEach(position => {
      const marketData = state.marketData.get(position.symbol);
      if (marketData) {
        totalValue += position.quantity * marketData.last;
      }
    });
    
    return totalValue;
  },
  
  getTopMovers: (limit = 10) => {
    const state = get();
    return Array.from(state.marketData.values())
      .sort((a: MarketDataPoint, b: MarketDataPoint) => Math.abs(b.changePercent) - Math.abs(a.changePercent))
      .slice(0, limit);
  }
}));

// Specialized selectors for optimized subscriptions
export const useMarketData = (symbol: string) => 
  useTradingStore(state => state.marketData.get(symbol));

export const usePositions = () => 
  useTradingStore(state => state.positions);

export const useActivePortfolio = () => 
  useTradingStore(state => {
    const activeId = state.activePortfolio;
    return state.portfolios.find(p => p.id === activeId) || null;
  });

export const useUnreadNotifications = () => 
  useTradingStore(state => state.notifications.filter(n => !n.read));

export const useConnectionStatus = () => 
  useTradingStore(state => state.connectionStatus); 