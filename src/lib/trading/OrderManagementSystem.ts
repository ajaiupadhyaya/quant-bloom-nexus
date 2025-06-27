// @ts-ignore
import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import { Order } from '../store/TradingStore';

export interface SmartOrder extends Order {
  originalQuantity: number;
  executedQuantity: number;
  avgExecutionPrice: number;
  executionStrategy: ExecutionStrategy;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY' | 'GTD';
  expirationTime?: number;
  minQuantity?: number;
  displayQuantity?: number;
  isHidden: boolean;
  postOnly: boolean;
  reduceOnly: boolean;
  stopPrice?: number;
  trailAmount?: number;
  trailPercent?: number;
  pegOffset?: number;
  parentOrderId?: string;
  childOrders: string[];
  exchangeRouting: ExchangeRoute[];
  lastModified: number;
  commissions: number;
  fees: number;
  orderTags: string[];
  clientOrderId: string;
  brokerOrderId?: string;
}

export interface ExecutionStrategy {
  type: 'TWAP' | 'VWAP' | 'POV' | 'IS' | 'ARRIVAL_PRICE' | 'MARKET' | 'LIMIT' | 'ICEBERG';
  startTime?: number;
  endTime?: number;
  participationRate?: number; // 0-1 for POV
  aggressiveness?: number; // 0-1 scale
  sliceSize?: number;
  maxSlices?: number;
  minInterval?: number; // milliseconds
  maxInterval?: number;
  priceImprovement?: boolean;
  darkPool?: boolean;
  adaptiveParameters?: boolean;
}

export interface ExchangeRoute {
  exchange: string;
  priority: number;
  allocation: number; // percentage 0-100
  isActive: boolean;
  latency: number;
  fees: number;
  liquidity: number;
  marketShare: number;
}

export interface ExecutionReport {
  orderId: string;
  executionId: string;
  timestamp: number;
  side: 'buy' | 'sell';
  symbol: string;
  quantity: number;
  price: number;
  value: number;
  exchange: string;
  commissions: number;
  fees: number;
  liquidity: 'added' | 'removed';
  executionType: 'new' | 'partial' | 'fill' | 'cancel' | 'replace' | 'reject';
  lastQuantity: number;
  lastPrice: number;
  leavesQuantity: number;
  avgPrice: number;
  orderStatus: 'new' | 'partially_filled' | 'filled' | 'cancelled' | 'rejected' | 'pending_cancel' | 'pending_replace';
}

export interface RiskRule {
  id: string;
  name: string;
  type: 'pre_trade' | 'post_trade' | 'position' | 'exposure';
  isActive: boolean;
  parameters: {
    maxOrderSize?: number;
    maxPositionSize?: number;
    maxDayLoss?: number;
    maxExposure?: number;
    allowedSymbols?: string[];
    restrictedSymbols?: string[];
    maxLeverage?: number;
    concentrationLimit?: number;
    velocityLimit?: { orders: number; timeWindow: number };
    priceDeviationLimit?: number;
  };
  action: 'reject' | 'limit' | 'warn' | 'hold';
  priority: number;
}

export interface OrderBook {
  symbol: string;
  timestamp: number;
  bids: Array<{ price: number; size: number; count: number }>;
  asks: Array<{ price: number; size: number; count: number }>;
  spread: number;
  midPrice: number;
  totalBidSize: number;
  totalAskSize: number;
  imbalance: number;
}

export interface TradingSession {
  id: string;
  name: string;
  startTime: number;
  endTime: number;
  isActive: boolean;
  exchanges: string[];
  allowedOrderTypes: string[];
  riskLimits: RiskRule[];
}

export class OrderManagementSystem extends (EventEmitter as any) {
  private orders: Map<string, SmartOrder> = new Map();
  private executionReports: Map<string, ExecutionReport[]> = new Map();
  private riskRules: Map<string, RiskRule> = new Map();
  private exchangeRoutes: Map<string, ExchangeRoute[]> = new Map();
  private orderBooks: Map<string, OrderBook> = new Map();
  private tradingSessions: Map<string, TradingSession> = new Map();
  private activeStrategies: Map<string, any> = new Map();
  
  // Performance tracking
  private metrics = {
    ordersSubmitted: 0,
    ordersFilled: 0,
    ordersRejected: 0,
    totalVolume: 0,
    avgFillTime: 0,
    avgSlippage: 0,
    implementationShortfall: 0,
    averageCommissions: 0,
    fillRate: 0
  };

  constructor() {
    super();
    this.initializeDefaultRiskRules();
    this.initializeExchangeRoutes();
    this.initializeTradingSessions();
    this.startOrderProcessor();
  }

  private initializeDefaultRiskRules(): void {
    const defaultRules: RiskRule[] = [
      {
        id: 'max_order_size',
        name: 'Maximum Order Size',
        type: 'pre_trade',
        isActive: true,
        parameters: { maxOrderSize: 1000000 },
        action: 'reject',
        priority: 1
      },
      {
        id: 'max_position_size',
        name: 'Maximum Position Size',
        type: 'position',
        isActive: true,
        parameters: { maxPositionSize: 5000000 },
        action: 'reject',
        priority: 2
      },
      {
        id: 'max_day_loss',
        name: 'Maximum Daily Loss',
        type: 'post_trade',
        isActive: true,
        parameters: { maxDayLoss: -100000 },
        action: 'hold',
        priority: 3
      },
      {
        id: 'velocity_limit',
        name: 'Order Velocity Limit',
        type: 'pre_trade',
        isActive: true,
        parameters: { velocityLimit: { orders: 100, timeWindow: 60000 } },
        action: 'limit',
        priority: 4
      }
    ];

    defaultRules.forEach(rule => this.riskRules.set(rule.id, rule));
  }

  private initializeExchangeRoutes(): void {
    const exchanges = [
      { name: 'NYSE', priority: 1, latency: 0.5, fees: 0.003, liquidity: 0.95, marketShare: 0.25 },
      { name: 'NASDAQ', priority: 2, latency: 0.6, fees: 0.0025, liquidity: 0.92, marketShare: 0.20 },
      { name: 'BATS', priority: 3, latency: 0.4, fees: 0.002, liquidity: 0.88, marketShare: 0.15 },
      { name: 'ARCA', priority: 4, latency: 0.7, fees: 0.0035, liquidity: 0.85, marketShare: 0.12 },
      { name: 'IEX', priority: 5, latency: 0.35, fees: 0.0009, liquidity: 0.75, marketShare: 0.08 }
    ];

    exchanges.forEach(exchange => {
      const route: ExchangeRoute = {
        exchange: exchange.name,
        priority: exchange.priority,
        allocation: exchange.marketShare * 100,
        isActive: true,
        latency: exchange.latency,
        fees: exchange.fees,
        liquidity: exchange.liquidity,
        marketShare: exchange.marketShare
      };
      
      const routes = this.exchangeRoutes.get('default') || [];
      routes.push(route);
      this.exchangeRoutes.set('default', routes);
    });
  }

  private initializeTradingSessions(): void {
    const sessions: TradingSession[] = [
      {
        id: 'regular',
        name: 'Regular Trading Hours',
        startTime: 9.5 * 60 * 60 * 1000, // 9:30 AM
        endTime: 16 * 60 * 60 * 1000, // 4:00 PM
        isActive: true,
        exchanges: ['NYSE', 'NASDAQ', 'BATS', 'ARCA'],
        allowedOrderTypes: ['market', 'limit', 'stop', 'stop_limit'],
        riskLimits: Array.from(this.riskRules.values())
      },
      {
        id: 'extended',
        name: 'Extended Hours Trading',
        startTime: 4 * 60 * 60 * 1000, // 4:00 AM
        endTime: 20 * 60 * 60 * 1000, // 8:00 PM
        isActive: true,
        exchanges: ['NASDAQ', 'ARCA'],
        allowedOrderTypes: ['limit'],
        riskLimits: Array.from(this.riskRules.values()).filter(r => r.type === 'pre_trade')
      }
    ];

    sessions.forEach(session => this.tradingSessions.set(session.id, session));
  }

  private startOrderProcessor(): void {
    // Process orders every 100ms
    setInterval(() => {
      this.processOrders();
    }, 100);

    // Update market data every 50ms
    setInterval(() => {
      this.updateMarketData();
    }, 50);

    // Risk monitoring every 1 second
    setInterval(() => {
      this.monitorRisk();
    }, 1000);
  }

  // Public API Methods

  public async submitOrder(orderRequest: Partial<SmartOrder>): Promise<{ success: boolean; orderId?: string; error?: string }> {
    try {
      // Generate order ID
      const orderId = uuidv4();
      const clientOrderId = orderRequest.clientOrderId || uuidv4();

      // Create smart order
      const order: SmartOrder = {
        id: orderId,
        clientOrderId,
        symbol: orderRequest.symbol!,
        side: orderRequest.side!,
        quantity: orderRequest.quantity!,
        originalQuantity: orderRequest.quantity!,
        executedQuantity: 0,
        price: orderRequest.price!,
        avgExecutionPrice: 0,
        status: 'pending',
        timestamp: Date.now(),
        lastModified: Date.now(),
        executionStrategy: orderRequest.executionStrategy || { type: 'LIMIT' },
        timeInForce: orderRequest.timeInForce || 'GTC',
        isHidden: orderRequest.isHidden || false,
        postOnly: orderRequest.postOnly || false,
        reduceOnly: orderRequest.reduceOnly || false,
        exchangeRouting: this.exchangeRoutes.get('default') || [],
        childOrders: [],
        commissions: 0,
        fees: 0,
        orderTags: orderRequest.orderTags || []
      };

      // Pre-trade risk check
      const riskCheck = await this.performRiskCheck(order, 'pre_trade');
      if (!riskCheck.passed) {
        return { success: false, error: riskCheck.reason };
      }

      // Store order
      this.orders.set(orderId, order);
      this.metrics.ordersSubmitted++;

      // Route order for execution
      await this.routeOrder(order);

      this.emit('order_submitted', { orderId, order });

      return { success: true, orderId };

    } catch (error) {
      console.error('Order submission failed:', error);
      return { success: false, error: 'Order submission failed' };
    }
  }

  public async cancelOrder(orderId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const order = this.orders.get(orderId);
      if (!order) {
        return { success: false, error: 'Order not found' };
      }

      if (order.status === 'filled' || order.status === 'cancelled') {
        return { success: false, error: `Cannot cancel order in ${order.status} status` };
      }

      order.status = 'cancelled';
      order.lastModified = Date.now();

      // Cancel child orders
      for (const childOrderId of order.childOrders) {
        await this.cancelOrder(childOrderId);
      }

      this.emit('order_cancelled', { orderId, order });

      return { success: true };

    } catch (error) {
      console.error('Order cancellation failed:', error);
      return { success: false, error: 'Order cancellation failed' };
    }
  }

  public async modifyOrder(
    orderId: string, 
    modifications: Partial<Pick<SmartOrder, 'quantity' | 'price' | 'timeInForce'>>
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const order = this.orders.get(orderId);
      if (!order) {
        return { success: false, error: 'Order not found' };
      }

      if (order.status !== 'pending' && order.status !== 'partially_filled') {
        return { success: false, error: `Cannot modify order in ${order.status} status` };
      }

      // Apply modifications
      if (modifications.quantity !== undefined) order.quantity = modifications.quantity;
      if (modifications.price !== undefined) order.price = modifications.price;
      if (modifications.timeInForce !== undefined) order.timeInForce = modifications.timeInForce;
      
      order.lastModified = Date.now();

      this.emit('order_modified', { orderId, order, modifications });

      return { success: true };

    } catch (error) {
      console.error('Order modification failed:', error);
      return { success: false, error: 'Order modification failed' };
    }
  }

  public getOrder(orderId: string): SmartOrder | null {
    return this.orders.get(orderId) || null;
  }

  public getOrdersBySymbol(symbol: string): SmartOrder[] {
    return Array.from(this.orders.values()).filter(order => order.symbol === symbol);
  }

  public getActiveOrders(): SmartOrder[] {
    return Array.from(this.orders.values()).filter(order => 
      order.status === 'pending' || order.status === 'partially_filled'
    );
  }

  public getExecutionReports(orderId: string): ExecutionReport[] {
    return this.executionReports.get(orderId) || [];
  }

  public getOrderMetrics() {
    return { ...this.metrics };
  }

  // Private Methods

  private async performRiskCheck(order: SmartOrder, checkType: 'pre_trade' | 'post_trade'): Promise<{ passed: boolean; reason?: string }> {
    const applicableRules = Array.from(this.riskRules.values())
      .filter(rule => rule.isActive && rule.type === checkType)
      .sort((a, b) => a.priority - b.priority);

    for (const rule of applicableRules) {
      const result = await this.evaluateRiskRule(rule, order);
      if (!result.passed) {
        if (rule.action === 'reject') {
          return { passed: false, reason: result.reason };
        } else if (rule.action === 'warn') {
          this.emit('risk_warning', { rule, order, reason: result.reason });
        } else if (rule.action === 'hold') {
          this.emit('risk_hold', { rule, order, reason: result.reason });
        }
      }
    }

    return { passed: true };
  }

  private async evaluateRiskRule(rule: RiskRule, order: SmartOrder): Promise<{ passed: boolean; reason?: string }> {
    const params = rule.parameters;

    switch (rule.id) {
      case 'max_order_size':
        if (params.maxOrderSize && order.quantity * order.price > params.maxOrderSize) {
          return { passed: false, reason: `Order size exceeds maximum: ${params.maxOrderSize}` };
        }
        break;

      case 'velocity_limit':
        if (params.velocityLimit) {
          const recentOrders = this.getRecentOrders(params.velocityLimit.timeWindow);
          if (recentOrders.length >= params.velocityLimit.orders) {
            return { passed: false, reason: `Order velocity limit exceeded: ${params.velocityLimit.orders} orders in ${params.velocityLimit.timeWindow}ms` };
          }
        }
        break;

      case 'max_position_size':
        if (params.maxPositionSize) {
          const currentPosition = this.getCurrentPosition(order.symbol);
          const newPosition = order.side === 'buy' ? 
            currentPosition + order.quantity : 
            currentPosition - order.quantity;
          
          if (Math.abs(newPosition * order.price) > params.maxPositionSize) {
            return { passed: false, reason: `Position size would exceed maximum: ${params.maxPositionSize}` };
          }
        }
        break;

      // Additional risk rule evaluations...
    }

    return { passed: true };
  }

  private async routeOrder(order: SmartOrder): Promise<void> {
    // Implement smart order routing based on execution strategy
    switch (order.executionStrategy.type) {
      case 'MARKET':
        await this.executeMarketOrder(order);
        break;
      case 'LIMIT':
        await this.executeLimitOrder(order);
        break;
      case 'TWAP':
        await this.executeTWAPOrder(order);
        break;
      case 'VWAP':
        await this.executeVWAPOrder(order);
        break;
      case 'ICEBERG':
        await this.executeIcebergOrder(order);
        break;
      default:
        await this.executeLimitOrder(order);
    }
  }

  private async executeMarketOrder(order: SmartOrder): Promise<void> {
    // Simulate market order execution
    const orderBook = this.orderBooks.get(order.symbol);
    if (!orderBook) return;

    const executionPrice = order.side === 'buy' ? orderBook.asks[0]?.price : orderBook.bids[0]?.price;
    if (!executionPrice) return;

    await this.createExecution(order, order.quantity, executionPrice, 'NYSE');
  }

  private async executeLimitOrder(order: SmartOrder): Promise<void> {
    // Simulate limit order execution
    const orderBook = this.orderBooks.get(order.symbol);
    if (!orderBook) return;

    const canExecute = order.side === 'buy' ? 
      orderBook.asks[0]?.price <= order.price :
      orderBook.bids[0]?.price >= order.price;

    if (canExecute) {
      const executionPrice = order.side === 'buy' ? orderBook.asks[0].price : orderBook.bids[0].price;
      const executionQuantity = Math.min(order.quantity - order.executedQuantity, 
        order.side === 'buy' ? orderBook.asks[0].size : orderBook.bids[0].size);
      
      await this.createExecution(order, executionQuantity, executionPrice, 'NYSE');
    }
  }

  private async executeTWAPOrder(order: SmartOrder): Promise<void> {
    // Time-Weighted Average Price execution
    const strategy = order.executionStrategy;
    const duration = (strategy.endTime || Date.now() + 3600000) - (strategy.startTime || Date.now());
    const sliceSize = strategy.sliceSize || Math.floor(order.quantity / 10);
    const interval = strategy.minInterval || 60000; // 1 minute

    // Schedule order slices
    const numberOfSlices = Math.ceil(order.quantity / sliceSize);
    const timePerSlice = duration / numberOfSlices;

    for (let i = 0; i < numberOfSlices; i++) {
      setTimeout(async () => {
        const remainingQuantity = order.quantity - order.executedQuantity;
        const currentSliceSize = Math.min(sliceSize, remainingQuantity);
        
        if (currentSliceSize > 0 && order.status !== 'cancelled') {
          await this.executeLimitOrder({
            ...order,
            quantity: currentSliceSize,
            id: `${order.id}_slice_${i}`
          } as SmartOrder);
        }
      }, i * timePerSlice);
    }
  }

  private async executeVWAPOrder(order: SmartOrder): Promise<void> {
    // Volume-Weighted Average Price execution
    // Simplified implementation - would use historical volume patterns
    const orderBook = this.orderBooks.get(order.symbol);
    if (!orderBook) return;

    const participationRate = order.executionStrategy.participationRate || 0.1;
    const availableVolume = order.side === 'buy' ? orderBook.totalAskSize : orderBook.totalBidSize;
    const executionQuantity = Math.min(
      order.quantity - order.executedQuantity,
      Math.floor(availableVolume * participationRate)
    );

    if (executionQuantity > 0) {
      const executionPrice = order.side === 'buy' ? orderBook.asks[0].price : orderBook.bids[0].price;
      await this.createExecution(order, executionQuantity, executionPrice, 'NYSE');
    }
  }

  private async executeIcebergOrder(order: SmartOrder): Promise<void> {
    // Iceberg order execution - only show small portion
    const displayQuantity = order.displayQuantity || Math.floor(order.quantity / 10);
    const visibleOrder = {
      ...order,
      quantity: Math.min(displayQuantity, order.quantity - order.executedQuantity)
    };

    await this.executeLimitOrder(visibleOrder as SmartOrder);
  }

  private async createExecution(order: SmartOrder, quantity: number, price: number, exchange: string): Promise<void> {
    const executionId = uuidv4();
    const execution: ExecutionReport = {
      orderId: order.id,
      executionId,
      timestamp: Date.now(),
      side: order.side,
      symbol: order.symbol,
      quantity,
      price,
      value: quantity * price,
      exchange,
      commissions: this.calculateCommissions(quantity, price),
      fees: this.calculateFees(quantity, price, exchange),
      liquidity: Math.random() > 0.5 ? 'added' : 'removed',
      executionType: 'fill',
      lastQuantity: quantity,
      lastPrice: price,
      leavesQuantity: order.quantity - order.executedQuantity - quantity,
      avgPrice: ((order.avgExecutionPrice * order.executedQuantity) + (price * quantity)) / (order.executedQuantity + quantity),
      orderStatus: order.executedQuantity + quantity >= order.quantity ? 'filled' : 'partially_filled'
    };

    // Update order
    order.executedQuantity += quantity;
    order.avgExecutionPrice = execution.avgPrice;
    order.status = execution.orderStatus;
    order.commissions += execution.commissions;
    order.fees += execution.fees;
    order.lastModified = Date.now();

    // Store execution report
    const reports = this.executionReports.get(order.id) || [];
    reports.push(execution);
    this.executionReports.set(order.id, reports);

    // Update metrics
    this.metrics.totalVolume += quantity * price;
    if (execution.orderStatus === 'filled') {
      this.metrics.ordersFilled++;
    }
    this.metrics.fillRate = this.metrics.ordersFilled / this.metrics.ordersSubmitted;
    this.metrics.averageCommissions = (this.metrics.averageCommissions + execution.commissions) / 2;

    this.emit('execution_report', execution);
    
    if (execution.orderStatus === 'filled') {
      this.emit('order_filled', { orderId: order.id, order });
    }
  }

  private calculateCommissions(quantity: number, price: number): number {
    // Simplified commission calculation
    const value = quantity * price;
    return Math.max(1, value * 0.0005); // 0.05% or $1 minimum
  }

  private calculateFees(quantity: number, price: number, exchange: string): number {
    // Exchange fees based on routing
    const routes = this.exchangeRoutes.get('default') || [];
    const route = routes.find(r => r.exchange === exchange);
    const feeRate = route?.fees || 0.003;
    
    return quantity * price * feeRate;
  }

  private processOrders(): void {
    // Process active orders
    const activeOrders = this.getActiveOrders();
    
    activeOrders.forEach(async (order) => {
      // Check for order expiration
      if (order.timeInForce === 'DAY' && this.isOrderExpired(order)) {
        await this.cancelOrder(order.id);
        return;
      }

      // Continue execution based on strategy
      if (order.executedQuantity < order.quantity) {
        await this.routeOrder(order);
      }
    });
  }

  private updateMarketData(): void {
    // Simulate market data updates
    this.orders.forEach((order) => {
      if (!this.orderBooks.has(order.symbol)) {
        // Generate simulated order book
        const midPrice = order.price;
        const spread = midPrice * 0.001; // 0.1% spread
        
        this.orderBooks.set(order.symbol, {
          symbol: order.symbol,
          timestamp: Date.now(),
          bids: [
            { price: midPrice - spread/2, size: 1000, count: 5 },
            { price: midPrice - spread, size: 2000, count: 8 }
          ],
          asks: [
            { price: midPrice + spread/2, size: 1200, count: 6 },
            { price: midPrice + spread, size: 1800, count: 7 }
          ],
          spread: spread,
          midPrice: midPrice,
          totalBidSize: 3000,
          totalAskSize: 3000,
          imbalance: 0
        });
      }
    });
  }

  private monitorRisk(): void {
    // Continuous risk monitoring
    const activeOrders = this.getActiveOrders();
    
    activeOrders.forEach(async (order) => {
      const riskCheck = await this.performRiskCheck(order, 'post_trade');
      if (!riskCheck.passed) {
        this.emit('risk_violation', { order, reason: riskCheck.reason });
        // Take appropriate action based on risk rule
      }
    });
  }

  private getRecentOrders(timeWindow: number): SmartOrder[] {
    const cutoff = Date.now() - timeWindow;
    return Array.from(this.orders.values()).filter(order => order.timestamp >= cutoff);
  }

  private getCurrentPosition(symbol: string): number {
    // Calculate current position from executed orders
    let position = 0;
    const symbolOrders = this.getOrdersBySymbol(symbol);
    
    symbolOrders.forEach(order => {
      if (order.status === 'filled') {
        const quantity = order.side === 'buy' ? order.executedQuantity : -order.executedQuantity;
        position += quantity;
      }
    });
    
    return position;
  }

  private isOrderExpired(order: SmartOrder): boolean {
    if (order.timeInForce === 'DAY') {
      const now = new Date();
      const orderDate = new Date(order.timestamp);
      return now.getDate() !== orderDate.getDate();
    }
    
    if (order.timeInForce === 'GTD' && order.expirationTime) {
      return Date.now() > order.expirationTime;
    }
    
    return false;
  }

  // Configuration Methods
  public addRiskRule(rule: RiskRule): void {
    this.riskRules.set(rule.id, rule);
  }

  public updateRiskRule(ruleId: string, updates: Partial<RiskRule>): void {
    const rule = this.riskRules.get(ruleId);
    if (rule) {
      Object.assign(rule, updates);
    }
  }

  public removeRiskRule(ruleId: string): void {
    this.riskRules.delete(ruleId);
  }

  public updateExchangeRouting(symbol: string, routes: ExchangeRoute[]): void {
    this.exchangeRoutes.set(symbol, routes);
  }

  public getSystemStatus(): any {
    return {
      totalOrders: this.orders.size,
      activeOrders: this.getActiveOrders().length,
      riskRules: this.riskRules.size,
      exchanges: this.exchangeRoutes.size,
      metrics: this.metrics,
      uptime: process.uptime(),
      timestamp: Date.now()
    };
  }
}

// Singleton instance
export const orderManagementSystem = new OrderManagementSystem(); 