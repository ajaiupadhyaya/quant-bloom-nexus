import { HistoricalDataPoint } from '../data/RealTimeDataProvider';
import { TechnicalIndicators } from './TechnicalIndicators';
import { PredictionModels } from '../ai/PredictionModels';

export interface BacktestConfig {
  initialCapital: number;
  startDate: number;
  endDate: number;
  commission: number;
  slippage: number;
  maxPositionSize: number;
  riskFreeRate: number;
  benchmark?: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: number;
  commission: number;
  slippage: number;
  pnl?: number;
  holdingPeriod?: number;
  strategy: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  entryTimestamp: number;
  marketValue: number;
}

export interface BacktestResults {
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  winRate: number;
  profitFactor: number;
  trades: Trade[];
  equity: Array<{ timestamp: number; value: number }>;
  positions: Position[];
  performance: PerformanceMetrics;
}

export interface PerformanceMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageWin: number;
  averageLoss: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  averageHoldingPeriod: number;
  turnover: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  trackingError: number;
}

export interface TradingStrategy {
  name: string;
  description: string;
  parameters: { [key: string]: any };
  initialize?: (data: HistoricalDataPoint[]) => void;
  generateSignals: (data: HistoricalDataPoint[], index: number) => TradingSignal[];
  riskManagement?: (position: Position, currentPrice: number) => RiskAction;
}

export interface TradingSignal {
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  quantity: number;
  price: number;
  timestamp: number;
  confidence: number;
  reason: string;
  stopLoss?: number;
  takeProfit?: number;
}

export interface RiskAction {
  action: 'hold' | 'reduce' | 'close';
  newQuantity?: number;
  reason: string;
}

export class BacktestingEngine {
  private config: BacktestConfig;
  private currentCapital: number;
  private position: number = 0;
  private trades: Trade[] = [];
  private positions: Map<string, Position> = new Map();
  private equity: Array<{ timestamp: number; value: number }> = [];
  private tradeId = 0;

  constructor(config: BacktestConfig) {
    this.config = config;
    this.currentCapital = config.initialCapital;
  }

  public async runBacktest(
    data: HistoricalDataPoint[],
    strategy: TradingStrategy
  ): Promise<BacktestResults> {
    this.reset();
    
    // Initialize strategy
    if (strategy.initialize) {
      strategy.initialize(data);
    }

    // Filter data by date range
    const filteredData = data.filter(
      point => point.timestamp >= this.config.startDate && 
               point.timestamp <= this.config.endDate
    );

    // Run simulation
    for (let i = 0; i < filteredData.length; i++) {
      const currentData = filteredData[i];
      
      // Update positions with current prices
      this.updatePositions(currentData);
      
      // Generate trading signals
      const signals = strategy.generateSignals(filteredData.slice(0, i + 1), i);
      
      // Execute trades
      for (const signal of signals) {
        this.executeSignal(signal, currentData);
      }
      
      // Apply risk management
      if (strategy.riskManagement) {
        this.applyRiskManagement(strategy.riskManagement, currentData);
      }
      
      // Record equity value
      this.recordEquity(currentData.timestamp);
    }

    return this.calculateResults(strategy.name);
  }

  private reset(): void {
    this.currentCapital = this.config.initialCapital;
    this.position = 0;
    this.trades = [];
    this.positions.clear();
    this.equity = [];
    this.tradeId = 0;
  }

  private recordEquity(data: HistoricalDataPoint): void {
    const positionValue = this.position * data.close;
    this.equity.push({
      timestamp: data.timestamp,
      value: this.currentCapital + positionValue
    });
  }

  private executeSignal(signal: TradingSignal, data: HistoricalDataPoint): void {
    if (signal.action === 'hold') return;

    const commission = signal.quantity * signal.price * this.config.commission;
    
    if (signal.action === 'buy' && this.position === 0) {
      const totalCost = signal.quantity * signal.price + commission;
      if (totalCost <= this.currentCapital) {
        this.currentCapital -= totalCost;
        this.position = signal.quantity;
        
        this.trades.push({
          id: `trade_${++this.tradeId}`,
          symbol: signal.symbol,
          side: 'buy',
          quantity: signal.quantity,
          price: signal.price,
          timestamp: signal.timestamp,
          commission,
          slippage: 0,
          strategy: 'backtest'
        });
      }
    } else if (signal.action === 'sell' && this.position > 0) {
      const proceeds = this.position * signal.price - commission;
      const buyTrade = this.trades.find(t => t.side === 'buy' && !t.pnl);
      
      if (buyTrade) {
        const pnl = proceeds - (buyTrade.quantity * buyTrade.price);
        buyTrade.pnl = pnl;
      }
      
      this.currentCapital += proceeds;
      this.position = 0;
      
      this.trades.push({
        id: `trade_${++this.tradeId}`,
        symbol: signal.symbol,
        side: 'sell',
        quantity: this.position,
        price: signal.price,
        timestamp: signal.timestamp,
        commission,
        slippage: 0,
        pnl: proceeds - (this.position * signal.price),
        holdingPeriod: signal.timestamp - (buyTrade?.timestamp || 0),
        strategy: 'backtest'
      });
    }
  }

  private applyRiskManagement(
    riskFunction: (position: Position, currentPrice: number) => RiskAction,
    data: HistoricalDataPoint
  ): void {
    this.positions.forEach((position, symbol) => {
      const action = riskFunction(position, data.close);
      
      if (action.action === 'close') {
        const signal: TradingSignal = {
          symbol,
          action: 'sell',
          quantity: position.quantity,
          price: data.close,
          timestamp: data.timestamp,
          confidence: 1,
          reason: action.reason
        };
        this.executeSignal(signal, data);
      } else if (action.action === 'reduce' && action.newQuantity) {
        const sellQuantity = position.quantity - action.newQuantity;
        const signal: TradingSignal = {
          symbol,
          action: 'sell',
          quantity: sellQuantity,
          price: data.close,
          timestamp: data.timestamp,
          confidence: 1,
          reason: action.reason
        };
        this.executeSignal(signal, data);
      }
    });
  }

  private calculateResults(strategyName: string): BacktestResults {
    const performance = this.calculatePerformanceMetrics();
    const equity = this.equity;
    
    const totalReturn = (equity[equity.length - 1].value - this.config.initialCapital) / this.config.initialCapital;
    const annualizedReturn = this.calculateAnnualizedReturn(equity);
    const volatility = this.calculateVolatility(equity);
    const sharpeRatio = (annualizedReturn - this.config.riskFreeRate) / volatility;
    const maxDrawdown = this.calculateMaxDrawdown(equity);
    
    return {
      totalReturn,
      annualizedReturn,
      volatility,
      sharpeRatio,
      sortinoRatio: this.calculateSortinoRatio(equity),
      calmarRatio: annualizedReturn / Math.abs(maxDrawdown.maxDrawdown),
      maxDrawdown: maxDrawdown.maxDrawdown,
      maxDrawdownDuration: maxDrawdown.maxDrawdownDuration,
      winRate: performance.winningTrades / performance.totalTrades,
      profitFactor: this.calculateProfitFactor(),
      trades: this.trades,
      equity,
      positions: Array.from(this.positions.values()),
      performance
    };
  }

  private calculatePerformanceMetrics(): PerformanceMetrics {
    const winningTrades = this.trades.filter(t => (t.pnl || 0) > 0);
    const losingTrades = this.trades.filter(t => (t.pnl || 0) < 0);
    
    return {
      totalTrades: this.trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      averageWin: winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / winningTrades.length || 0,
      averageLoss: losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / losingTrades.length || 0,
      largestWin: Math.max(...winningTrades.map(t => t.pnl || 0), 0),
      largestLoss: Math.min(...losingTrades.map(t => t.pnl || 0), 0),
      consecutiveWins: this.calculateConsecutiveWins(),
      consecutiveLosses: this.calculateConsecutiveLosses(),
      averageHoldingPeriod: this.trades.reduce((sum, t) => sum + (t.holdingPeriod || 0), 0) / this.trades.length || 0,
      turnover: this.calculateTurnover(),
      beta: 0, // Would need benchmark data
      alpha: 0, // Would need benchmark data
      informationRatio: 0, // Would need benchmark data
      trackingError: 0 // Would need benchmark data
    };
  }

  // Helper methods
  private createEmptyPosition(symbol: string): Position {
    return {
      symbol,
      quantity: 0,
      averagePrice: 0,
      currentPrice: 0,
      unrealizedPnL: 0,
      realizedPnL: 0,
      entryTimestamp: 0,
      marketValue: 0
    };
  }

  private calculateCommission(orderValue: number): number {
    return orderValue * this.config.commission;
  }

  private calculateSlippage(price: number, quantity: number): number {
    return price * quantity * this.config.slippage;
  }

  private calculateAnnualizedReturn(equity: Array<{ timestamp: number; value: number }>): number {
    if (equity.length < 2) return 0;
    
    const startValue = equity[0].value;
    const endValue = equity[equity.length - 1].value;
    const startTime = equity[0].timestamp;
    const endTime = equity[equity.length - 1].timestamp;
    
    const years = (endTime - startTime) / (365.25 * 24 * 60 * 60 * 1000);
    return Math.pow(endValue / startValue, 1 / years) - 1;
  }

  private calculateVolatility(equity: Array<{ timestamp: number; value: number }>): number {
    const returns = [];
    for (let i = 1; i < equity.length; i++) {
      const returnValue = (equity[i].value - equity[i - 1].value) / equity[i - 1].value;
      returns.push(returnValue);
    }
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance * 252); // Annualized
  }

  private calculateMaxDrawdown(equity: Array<{ timestamp: number; value: number }>): {
    maxDrawdown: number;
    maxDrawdownDuration: number;
  } {
    let maxDrawdown = 0;
    let maxDrawdownDuration = 0;
    let peak = equity[0].value;
    let drawdownStart = 0;
    
    for (let i = 1; i < equity.length; i++) {
      if (equity[i].value > peak) {
        peak = equity[i].value;
        drawdownStart = i;
      } else {
        const drawdown = (peak - equity[i].value) / peak;
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown;
          maxDrawdownDuration = equity[i].timestamp - equity[drawdownStart].timestamp;
        }
      }
    }
    
    return { maxDrawdown, maxDrawdownDuration };
  }

  private calculateSortinoRatio(equity: Array<{ timestamp: number; value: number }>): number {
    const returns = [];
    for (let i = 1; i < equity.length; i++) {
      returns.push((equity[i].value - equity[i - 1].value) / equity[i - 1].value);
    }
    
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const downside = returns.filter(r => r < 0);
    const downsideDeviation = Math.sqrt(
      downside.reduce((sum, r) => sum + r * r, 0) / downside.length
    );
    
    return (meanReturn * 252 - this.config.riskFreeRate) / (downsideDeviation * Math.sqrt(252));
  }

  private calculateProfitFactor(): number {
    const grossProfit = this.trades
      .filter(t => (t.pnl || 0) > 0)
      .reduce((sum, t) => sum + (t.pnl || 0), 0);
    
    const grossLoss = Math.abs(
      this.trades
        .filter(t => (t.pnl || 0) < 0)
        .reduce((sum, t) => sum + (t.pnl || 0), 0)
    );
    
    return grossLoss > 0 ? grossProfit / grossLoss : 0;
  }

  private calculateConsecutiveWins(): number {
    let maxConsecutive = 0;
    let current = 0;
    
    for (const trade of this.trades) {
      if ((trade.pnl || 0) > 0) {
        current++;
        maxConsecutive = Math.max(maxConsecutive, current);
      } else {
        current = 0;
      }
    }
    
    return maxConsecutive;
  }

  private calculateConsecutiveLosses(): number {
    let maxConsecutive = 0;
    let current = 0;
    
    for (const trade of this.trades) {
      if ((trade.pnl || 0) < 0) {
        current++;
        maxConsecutive = Math.max(maxConsecutive, current);
      } else {
        current = 0;
      }
    }
    
    return maxConsecutive;
  }

  private calculateTurnover(): number {
    const totalTradeValue = this.trades.reduce((sum, t) => sum + (t.quantity * t.price), 0);
    const avgPortfolioValue = this.equity.reduce((sum, e) => sum + e.value, 0) / this.equity.length;
    return totalTradeValue / avgPortfolioValue;
  }

  // Pre-built strategies
  public static getMACDStrategy(): TradingStrategy {
    return {
      name: 'MACD Crossover',
      description: 'Buy when MACD crosses above signal line, sell when below',
      parameters: { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 },
      generateSignals: (data: HistoricalDataPoint[], index: number) => {
        if (index < 50) return []; // Need enough data for indicators
        
        const indicators = TechnicalIndicators.calculateMACD(data.slice(0, index + 1));
        if (indicators.length < 2) return [];
        
        const current = indicators[indicators.length - 1];
        const previous = indicators[indicators.length - 2];
        
        const signals: TradingSignal[] = [];
        
        if (previous.macd <= previous.signal && current.macd > current.signal) {
          signals.push({
            symbol: 'DEFAULT',
            action: 'buy',
            quantity: 100,
            price: data[index].close,
            timestamp: data[index].timestamp,
            confidence: 0.7,
            reason: 'MACD bullish crossover'
          });
        } else if (previous.macd >= previous.signal && current.macd < current.signal) {
          signals.push({
            symbol: 'DEFAULT',
            action: 'sell',
            quantity: 100,
            price: data[index].close,
            timestamp: data[index].timestamp,
            confidence: 0.7,
            reason: 'MACD bearish crossover'
          });
        }
        
        return signals;
      }
    };
  }

  public static getRSIStrategy(): TradingStrategy {
    return {
      name: 'RSI Mean Reversion',
      description: 'Buy when RSI is oversold, sell when overbought',
      parameters: { period: 14, oversold: 30, overbought: 70 },
      generateSignals: (data: HistoricalDataPoint[], index: number) => {
        if (index < 20) return [];
        
        const indicators = TechnicalIndicators.calculateRSI(data.slice(0, index + 1));
        if (indicators.length === 0) return [];
        
        const current = indicators[indicators.length - 1];
        const signals: TradingSignal[] = [];
        
        if (current.rsi < 30) {
          signals.push({
            symbol: 'DEFAULT',
            action: 'buy',
            quantity: 100,
            price: data[index].close,
            timestamp: data[index].timestamp,
            confidence: (30 - current.rsi) / 30,
            reason: `RSI oversold at ${current.rsi.toFixed(1)}`
          });
        } else if (current.rsi > 70) {
          signals.push({
            symbol: 'DEFAULT',
            action: 'sell',
            quantity: 100,
            price: data[index].close,
            timestamp: data[index].timestamp,
            confidence: (current.rsi - 70) / 30,
            reason: `RSI overbought at ${current.rsi.toFixed(1)}`
          });
        }
        
        return signals;
      }
    };
  }
} 