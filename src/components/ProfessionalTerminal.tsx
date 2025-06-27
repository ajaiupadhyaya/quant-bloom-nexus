import { useDashboard } from '@/context/DashboardContext';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { realTimeDataProvider, RealTimeQuote, HistoricalDataPoint } from '../lib/data/RealTimeDataProvider';
import { TechnicalIndicators } from '../lib/analysis/TechnicalIndicators';
import { PredictionModels, PredictionResult, MarketRegime } from '../lib/ai/PredictionModels';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, CandlestickChart, ReferenceLine } from 'recharts';

interface TerminalTab {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
}

interface AdvancedChart {
  symbol: string;
  timeframe: string;
  data: HistoricalDataPoint[];
  indicators: any;
  predictions: PredictionResult[];
}

const TERMINAL_TABS: TerminalTab[] = [
  { id: 'market', name: 'MARKET', description: 'Real-time market data & quotes', icon: '📊', color: 'text-orange-400' },
  { id: 'chart', name: 'CHART', description: 'Advanced charting & technical analysis', icon: '📈', color: 'text-green-400' },
  { id: 'ai', name: 'AI', description: 'Machine learning predictions & analysis', icon: '🤖', color: 'text-blue-400' },
  { id: 'portfolio', name: 'PORT', description: 'Portfolio management & analytics', icon: '💼', color: 'text-purple-400' },
  { id: 'research', name: 'RESEARCH', description: 'Fundamental analysis & research', icon: '🔍', color: 'text-yellow-400' },
  { id: 'screen', name: 'SCREEN', description: 'Stock screeners & scanners', icon: '🎯', color: 'text-red-400' },
  { id: 'analytics', name: 'ANALYTICS', description: 'Statistical analysis & modeling', icon: '📋', color: 'text-indigo-400' },
  { id: 'backtest', name: 'BACKTEST', description: 'Strategy backtesting & optimization', icon: '⚡', color: 'text-pink-400' },
  { id: 'risk', name: 'RISK', description: 'Risk management & scenario analysis', icon: '⚠️', color: 'text-amber-400' },
  { id: 'news', name: 'NEWS', description: 'Real-time news & sentiment analysis', icon: '📰', color: 'text-cyan-400' },
  { id: 'econ', name: 'ECON', description: 'Economic data & calendar', icon: '🏛️', color: 'text-emerald-400' },
  { id: 'options', name: 'OPTIONS', description: 'Options analysis & strategies', icon: '🎲', color: 'text-violet-400' }
];

const POPULAR_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'SPY', 'QQQ'];

export const ProfessionalTerminal: React.FC = () => {
  const [activeTab, setActiveTab] = useState('market');
  const { selectedSymbol, setSelectedSymbol } = useDashboard();
  const [marketData, setMarketData] = useState<Map<string, RealTimeQuote>>(new Map());
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);
  const [indicators, setIndicators] = useState<any>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [marketRegime, setMarketRegime] = useState<MarketRegime[]>([]);
  const [commandInput, setCommandInput] = useState('');
  const [notifications, setNotifications] = useState<string[]>([]);

  // Real-time data subscription
  useEffect(() => {
    const symbols = POPULAR_SYMBOLS;
    
    const handleQuoteUpdate = (quote: RealTimeQuote) => {
      setMarketData(prev => new Map(prev.set(quote.symbol, quote)));
      
      if (Math.abs(quote.changePercent) > 2) {
        setNotifications(prev => [
          `${quote.symbol}: ${quote.changePercent > 0 ? '+' : ''}${quote.changePercent.toFixed(2)}% - ${quote.price}`,
          ...prev.slice(0, 4)
        ]);
      }
    };

    realTimeDataProvider.on('quote', handleQuoteUpdate);
    realTimeDataProvider.subscribeToRealTimeData(symbols);

    return () => {
      realTimeDataProvider.off('quote', handleQuoteUpdate);
      realTimeDataProvider.unsubscribeFromRealTimeData(symbols);
    };
  }, []);

  // Load historical data and analysis for selected symbol
  useEffect(() => {
    const loadSymbolData = async () => {
      try {
        const historical = await realTimeDataProvider.getHistoricalData(selectedSymbol, '1y');
        setHistoricalData(historical);

        if (historical.length > 0) {
          const calculatedIndicators = TechnicalIndicators.calculateAllIndicators(historical);
          setIndicators(calculatedIndicators);

          const aiPredictions = await PredictionModels.predictPricesLSTM(historical, 30);
          setPredictions(aiPredictions);

          const regimeAnalysis = PredictionModels.detectMarketRegime(historical);
          setMarketRegime(regimeAnalysis);
        }
      } catch (error) {
        console.error('Error loading symbol data:', error);
      }
    };

    loadSymbolData();
  }, [selectedSymbol]);

  const handleCommand = useCallback((command: string) => {
    const parts = command.toLowerCase().split(' ');
    const cmd = parts[0];

    switch (cmd) {
      case 'chart':
        setActiveTab('chart');
        if (parts[1]) setSelectedSymbol(parts[1].toUpperCase());
        break;
      case 'ai':
        setActiveTab('ai');
        if (parts[1]) setSelectedSymbol(parts[1].toUpperCase());
        break;
      case 'port':
        setActiveTab('portfolio');
        break;
      case 'screen':
        setActiveTab('screen');
        break;
      case 'news':
        setActiveTab('news');
        break;
      default:
        if (cmd.length >= 3 && cmd.length <= 5) {
          setSelectedSymbol(cmd.toUpperCase());
          setActiveTab('market');
        }
    }
    setCommandInput('');
  }, []);

  const currentQuote = marketData.get(selectedSymbol);
  const currentRegime = marketRegime[marketRegime.length - 1];

  return (
    <div className="h-screen bg-black text-green-400 font-mono overflow-hidden">
      {/* Terminal Header */}
      <div className="h-16 bg-gray-900 border-b border-orange-500 px-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="text-orange-400 text-xl font-bold">BLOOMBERG TERMINAL</div>
          <div className="text-gray-400">|</div>
          <div className="text-white text-sm">
            {new Date().toLocaleString()} EST
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-orange-400 text-sm">
            {currentQuote && (
              <span className={currentQuote.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}>
                {selectedSymbol}: ${currentQuote.price.toFixed(2)} 
                ({currentQuote.changePercent >= 0 ? '+' : ''}{currentQuote.changePercent.toFixed(2)}%)
              </span>
            )}
          </div>
          
          <div className="flex space-x-2">
            {notifications.map((notification, index) => (
              <Badge key={index} variant="outline" className="text-orange-400 border-orange-500">
                {notification}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      {/* Symbol Selection Bar */}
      <div className="h-12 bg-gray-900 border-b border-gray-700 px-4 flex items-center space-x-4">
        <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
          <SelectTrigger className="w-32 bg-black border-orange-500 text-orange-400">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-black border-gray-700">
            {POPULAR_SYMBOLS.map(symbol => (
              <SelectItem key={symbol} value={symbol} className="text-green-400">
                {symbol}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Input
          value={commandInput}
          onChange={(e) => setCommandInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleCommand(commandInput)}
          placeholder="Enter command or symbol..."
          className="flex-1 bg-black border-gray-700 text-green-400 placeholder-gray-500"
        />

        {currentRegime && (
          <Badge 
            variant="outline" 
            className={`border-${currentRegime.regime === 'bull' ? 'green' : 
              currentRegime.regime === 'bear' ? 'red' : 
              currentRegime.regime === 'volatile' ? 'yellow' : 'blue'}-500`}
          >
            {currentRegime.regime.toUpperCase()} MARKET ({Math.round(currentRegime.confidence * 100)}%)
          </Badge>
        )}
      </div>

      {/* Main Terminal Interface */}
      <div className="flex-1 flex">
        {/* Tab Navigation */}
        <div className="w-48 bg-gray-900 border-r border-gray-700 overflow-y-auto">
          <div className="p-2">
            {TERMINAL_TABS.map(tab => (
              <Button
                key={tab.id}
                variant={activeTab === tab.id ? "default" : "ghost"}
                className={`w-full mb-1 justify-start text-left ${
                  activeTab === tab.id 
                    ? 'bg-orange-600 text-white' 
                    : 'text-gray-300 hover:text-white hover:bg-gray-800'
                }`}
                onClick={() => setActiveTab(tab.id)}
              >
                <span className="mr-2">{tab.icon}</span>
                <div>
                  <div className="font-semibold">{tab.name}</div>
                  <div className="text-xs opacity-70">{tab.description}</div>
                </div>
              </Button>
            ))}
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            
            {/* MARKET TAB */}
            <TabsContent value="market" className="p-4 space-y-4">
              <div className="grid grid-cols-4 gap-4">
                {POPULAR_SYMBOLS.map(symbol => {
                  const quote = marketData.get(symbol);
                  return (
                    <Card key={symbol} className="bg-gray-900 border-gray-700">
                      <CardContent className="p-3">
                        <div className="flex justify-between items-center">
                          <div className="text-orange-400 font-semibold">{symbol}</div>
                          <div className={`text-sm ${quote?.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {quote?.changePercent >= 0 ? '+' : ''}{quote?.changePercent.toFixed(2)}%
                          </div>
                        </div>
                        <div className="text-white text-lg font-bold">
                          ${quote?.price.toFixed(2) || '--'}
                        </div>
                        <div className="text-xs text-gray-400">
                          Vol: {quote?.volume.toLocaleString() || '--'}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </TabsContent>

            {/* CHART TAB */}
            <TabsContent value="chart" className="p-4">
              <Card className="bg-gray-900 border-gray-700 h-96">
                <CardHeader>
                  <CardTitle className="text-orange-400">{selectedSymbol} - Advanced Chart</CardTitle>
                </CardHeader>
                <CardContent>
                  {historicalData.length > 0 && (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={historicalData.slice(-100)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis 
                          dataKey="timestamp" 
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          stroke="#9CA3AF"
                        />
                        <YAxis stroke="#9CA3AF" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #4B5563' }}
                          labelStyle={{ color: '#F97316' }}
                        />
                        <Line type="monotone" dataKey="close" stroke="#10B981" strokeWidth={2} dot={false} />
                        {indicators?.sma20 && (
                          <Line type="monotone" dataKey="sma20" stroke="#F59E0B" strokeWidth={1} dot={false} />
                        )}
                        {indicators?.sma50 && (
                          <Line type="monotone" dataKey="sma50" stroke="#EF4444" strokeWidth={1} dot={false} />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* AI TAB */}
            <TabsContent value="ai" className="p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-gray-900 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-blue-400">AI Price Predictions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {predictions.slice(0, 5).map((pred, index) => (
                      <div key={index} className="flex justify-between items-center py-1">
                        <span className="text-gray-300">
                          {new Date(pred.timestamp).toLocaleDateString()}
                        </span>
                        <span className={`font-semibold ${pred.trend === 'bullish' ? 'text-green-400' : 
                          pred.trend === 'bearish' ? 'text-red-400' : 'text-yellow-400'}`}>
                          ${pred.predictedPrice.toFixed(2)} ({Math.round(pred.confidence * 100)}%)
                        </span>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="bg-gray-900 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-blue-400">Technical Signals</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {indicators && (
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>RSI:</span>
                          <span className={`font-semibold ${
                            indicators.rsi[indicators.rsi.length - 1]?.rsi > 70 ? 'text-red-400' :
                            indicators.rsi[indicators.rsi.length - 1]?.rsi < 30 ? 'text-green-400' :
                            'text-yellow-400'
                          }`}>
                            {indicators.rsi[indicators.rsi.length - 1]?.rsi.toFixed(1)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>MACD:</span>
                          <span className={`font-semibold ${
                            indicators.macd[indicators.macd.length - 1]?.crossover === 'bullish' ? 'text-green-400' :
                            indicators.macd[indicators.macd.length - 1]?.crossover === 'bearish' ? 'text-red-400' :
                            'text-yellow-400'
                          }`}>
                            {indicators.macd[indicators.macd.length - 1]?.crossover || 'NEUTRAL'}
                          </span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Additional tabs would be implemented similarly */}
            <TabsContent value="portfolio" className="p-4">
              <Card className="bg-gray-900 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-purple-400">Portfolio Management</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-gray-400">
                    Portfolio management features coming soon...
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Other tabs follow similar pattern */}
          </Tabs>
        </div>
      </div>

      {/* Status Bar */}
      <div className="h-8 bg-gray-900 border-t border-gray-700 px-4 flex items-center justify-between text-xs">
        <div className="text-gray-400">
          Data: Real-time | Symbols: {marketData.size} active | AI: {predictions.length} forecasts
        </div>
        <div className="text-orange-400">
          F8: Equity | F9: Bonds | F10: FX | F11: Commodities | F12: Crypto
        </div>
      </div>
    </div>
  );
}; 