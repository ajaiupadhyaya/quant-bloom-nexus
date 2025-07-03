import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Search, Filter, TrendingUp, TrendingDown, BarChart3, DollarSign, Target, Shield } from 'lucide-react';

interface ScreenerCriteria {
  // Fundamental Metrics
  marketCapMin: number;
  marketCapMax: number;
  peRatioMin: number;
  peRatioMax: number;
  pbRatioMin: number;
  pbRatioMax: number;
  debtToEquityMax: number;
  roePct: number;
  roaPct: number;
  profitMarginPct: number;
  
  // Technical Metrics
  priceMin: number;
  priceMax: number;
  volumeMin: number;
  rsiMin: number;
  rsiMax: number;
  sma20Above: boolean;
  sma50Above: boolean;
  
  // Growth Metrics
  revenueGrowthPct: number;
  earningsGrowthPct: number;
  
  // Dividend Metrics
  dividendYieldMin: number;
  dividendYieldMax: number;
  
  // Risk Metrics
  betaMin: number;
  betaMax: number;
  volatilityMax: number;
  
  // Sector/Industry
  sectors: string[];
  exchanges: string[];
}

interface ScreenerResult {
  symbol: string;
  name: string;
  price: number;
  marketCap: number;
  peRatio: number;
  pbRatio: number;
  debtToEquity: number;
  roe: number;
  roa: number;
  profitMargin: number;
  revenueGrowth: number;
  earningsGrowth: number;
  dividendYield: number;
  beta: number;
  volatility: number;
  rsi: number;
  volume: number;
  sector: string;
  industry: string;
  exchange: string;
  score: number;
}

export const AdvancedScreener: React.FC = () => {
  const [criteria, setCriteria] = useState<ScreenerCriteria>({
    marketCapMin: 0,
    marketCapMax: 1000000,
    peRatioMin: 0,
    peRatioMax: 50,
    pbRatioMin: 0,
    pbRatioMax: 10,
    debtToEquityMax: 2,
    roePct: 10,
    roaPct: 5,
    profitMarginPct: 5,
    priceMin: 1,
    priceMax: 1000,
    volumeMin: 100000,
    rsiMin: 30,
    rsiMax: 70,
    sma20Above: false,
    sma50Above: false,
    revenueGrowthPct: 0,
    earningsGrowthPct: 0,
    dividendYieldMin: 0,
    dividendYieldMax: 10,
    betaMin: 0,
    betaMax: 3,
    volatilityMax: 50,
    sectors: [],
    exchanges: []
  });

  const [results, setResults] = useState<ScreenerResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [sortBy, setSortBy] = useState<keyof ScreenerResult>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [activeFilters, setActiveFilters] = useState(0);

  const sectors = [
    'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
    'Communication Services', 'Industrials', 'Consumer Defensive', 'Energy',
    'Utilities', 'Real Estate', 'Basic Materials'
  ];

  const exchanges = ['NYSE', 'NASDAQ', 'AMEX'];

  const presetScreens = {
    'Value Stocks': {
      peRatioMax: 15,
      pbRatioMax: 2,
      debtToEquityMax: 1,
      roePct: 15,
      profitMarginPct: 10
    },
    'Growth Stocks': {
      revenueGrowthPct: 20,
      earningsGrowthPct: 25,
      peRatioMax: 30,
      roePct: 20
    },
    'Dividend Stocks': {
      dividendYieldMin: 3,
      dividendYieldMax: 8,
      peRatioMax: 20,
      debtToEquityMax: 1.5
    },
    'Momentum Stocks': {
      rsiMin: 50,
      rsiMax: 80,
      sma20Above: true,
      sma50Above: true,
      volumeMin: 1000000
    },
    'Low Volatility': {
      volatilityMax: 20,
      betaMax: 1,
      debtToEquityMax: 1
    },
    'Small Cap Growth': {
      marketCapMax: 2000,
      revenueGrowthPct: 15,
      earningsGrowthPct: 20,
      roePct: 15
    }
  };

  useEffect(() => {
    calculateActiveFilters();
  }, [criteria]);

  const calculateActiveFilters = () => {
    let count = 0;
    if (criteria.marketCapMin > 0 || criteria.marketCapMax < 1000000) count++;
    if (criteria.peRatioMin > 0 || criteria.peRatioMax < 50) count++;
    if (criteria.pbRatioMin > 0 || criteria.pbRatioMax < 10) count++;
    if (criteria.debtToEquityMax < 2) count++;
    if (criteria.roePct > 10) count++;
    if (criteria.rsiMin > 30 || criteria.rsiMax < 70) count++;
    if (criteria.sma20Above || criteria.sma50Above) count++;
    if (criteria.sectors.length > 0) count++;
    setActiveFilters(count);
  };

  const runScreen = async () => {
    setLoading(true);
    try {
      // Simulate API call - in real implementation, this would call the backend
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate mock results based on criteria
      const mockResults: ScreenerResult[] = generateMockResults();
      setResults(mockResults);
    } catch (error) {
      console.error('Screening failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateMockResults = (): ScreenerResult[] => {
    const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL'];
    const companies = [
      'Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation', 'Amazon.com Inc.',
      'Tesla Inc.', 'Meta Platforms Inc.', 'NVIDIA Corporation', 'Netflix Inc.',
      'Advanced Micro Devices', 'Salesforce Inc.', 'Adobe Inc.', 'PayPal Holdings'
    ];

    return symbols.map((symbol, index) => ({
      symbol,
      name: companies[index],
      price: 50 + Math.random() * 200,
      marketCap: 100 + Math.random() * 2000,
      peRatio: 10 + Math.random() * 30,
      pbRatio: 1 + Math.random() * 5,
      debtToEquity: Math.random() * 1.5,
      roe: 10 + Math.random() * 20,
      roa: 5 + Math.random() * 15,
      profitMargin: 5 + Math.random() * 25,
      revenueGrowth: -5 + Math.random() * 30,
      earningsGrowth: -10 + Math.random() * 40,
      dividendYield: Math.random() * 5,
      beta: 0.5 + Math.random() * 1.5,
      volatility: 15 + Math.random() * 25,
      rsi: 30 + Math.random() * 40,
      volume: 1000000 + Math.random() * 10000000,
      sector: sectors[Math.floor(Math.random() * sectors.length)],
      industry: 'Software',
      exchange: exchanges[Math.floor(Math.random() * exchanges.length)],
      score: 60 + Math.random() * 40
    })).filter(stock => {
      // Apply basic filtering logic
      return stock.marketCap >= criteria.marketCapMin &&
             stock.marketCap <= criteria.marketCapMax &&
             stock.peRatio >= criteria.peRatioMin &&
             stock.peRatio <= criteria.peRatioMax &&
             stock.roe >= criteria.roePct;
    }).sort((a, b) => b.score - a.score);
  };

  const applyPreset = (presetName: keyof typeof presetScreens) => {
    const preset = presetScreens[presetName];
    setCriteria(prev => ({ ...prev, ...preset }));
  };

  const resetCriteria = () => {
    setCriteria({
      marketCapMin: 0,
      marketCapMax: 1000000,
      peRatioMin: 0,
      peRatioMax: 50,
      pbRatioMin: 0,
      pbRatioMax: 10,
      debtToEquityMax: 2,
      roePct: 10,
      roaPct: 5,
      profitMarginPct: 5,
      priceMin: 1,
      priceMax: 1000,
      volumeMin: 100000,
      rsiMin: 30,
      rsiMax: 70,
      sma20Above: false,
      sma50Above: false,
      revenueGrowthPct: 0,
      earningsGrowthPct: 0,
      dividendYieldMin: 0,
      dividendYieldMax: 10,
      betaMin: 0,
      betaMax: 3,
      volatilityMax: 50,
      sectors: [],
      exchanges: []
    });
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals);
  };

  const formatCurrency = (num: number) => {
    if (num >= 1000) {
      return `$${(num / 1000).toFixed(1)}B`;
    }
    return `$${num.toFixed(1)}M`;
  };

  const formatPercentage = (num: number) => {
    return `${num.toFixed(1)}%`;
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-50';
    if (score >= 60) return 'text-blue-600 bg-blue-50';
    if (score >= 40) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const sortResults = (field: keyof ScreenerResult) => {
    const newOrder = sortBy === field && sortOrder === 'desc' ? 'asc' : 'desc';
    setSortBy(field);
    setSortOrder(newOrder);
    
    const sorted = [...results].sort((a, b) => {
      const aVal = a[field];
      const bVal = b[field];
      if (newOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      }
      return aVal < bVal ? 1 : -1;
    });
    setResults(sorted);
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900">Advanced Stock Screener</h2>
        <div className="flex items-center space-x-4">
          <Badge variant="outline" className="text-sm">
            <Filter className="h-4 w-4 mr-1" />
            {activeFilters} Active Filters
          </Badge>
          <Button onClick={resetCriteria} variant="outline">Reset All</Button>
          <Button onClick={runScreen} disabled={loading}>
            {loading ? 'Screening...' : 'Run Screen'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Screening Criteria */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Search className="h-5 w-5 mr-2" />
                Screening Criteria
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="fundamental" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="fundamental">Fundamental</TabsTrigger>
                  <TabsTrigger value="technical">Technical</TabsTrigger>
                  <TabsTrigger value="other">Other</TabsTrigger>
                </TabsList>

                <TabsContent value="fundamental" className="space-y-4">
                  {/* Market Cap */}
                  <div className="space-y-2">
                    <Label>Market Cap (Billions)</Label>
                    <div className="grid grid-cols-2 gap-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={criteria.marketCapMin}
                        onChange={(e) => setCriteria({...criteria, marketCapMin: parseFloat(e.target.value) || 0})}
                      />
                      <Input
                        type="number"
                        placeholder="Max"
                        value={criteria.marketCapMax}
                        onChange={(e) => setCriteria({...criteria, marketCapMax: parseFloat(e.target.value) || 1000000})}
                      />
                    </div>
                  </div>

                  {/* P/E Ratio */}
                  <div className="space-y-2">
                    <Label>P/E Ratio</Label>
                    <Slider
                      value={[criteria.peRatioMin, criteria.peRatioMax]}
                      onValueChange={(values) => setCriteria({...criteria, peRatioMin: values[0], peRatioMax: values[1]})}
                      max={50}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>{criteria.peRatioMin}</span>
                      <span>{criteria.peRatioMax}</span>
                    </div>
                  </div>

                  {/* P/B Ratio */}
                  <div className="space-y-2">
                    <Label>P/B Ratio</Label>
                    <Slider
                      value={[criteria.pbRatioMin, criteria.pbRatioMax]}
                      onValueChange={(values) => setCriteria({...criteria, pbRatioMin: values[0], pbRatioMax: values[1]})}
                      max={10}
                      step={0.1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>{criteria.pbRatioMin}</span>
                      <span>{criteria.pbRatioMax}</span>
                    </div>
                  </div>

                  {/* ROE */}
                  <div className="space-y-2">
                    <Label>ROE Minimum (%)</Label>
                    <Input
                      type="number"
                      value={criteria.roePct}
                      onChange={(e) => setCriteria({...criteria, roePct: parseFloat(e.target.value) || 0})}
                    />
                  </div>

                  {/* Debt to Equity */}
                  <div className="space-y-2">
                    <Label>Max Debt/Equity</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={criteria.debtToEquityMax}
                      onChange={(e) => setCriteria({...criteria, debtToEquityMax: parseFloat(e.target.value) || 0})}
                    />
                  </div>

                  {/* Profit Margin */}
                  <div className="space-y-2">
                    <Label>Min Profit Margin (%)</Label>
                    <Input
                      type="number"
                      value={criteria.profitMarginPct}
                      onChange={(e) => setCriteria({...criteria, profitMarginPct: parseFloat(e.target.value) || 0})}
                    />
                  </div>

                  {/* Growth Metrics */}
                  <div className="space-y-2">
                    <Label>Min Revenue Growth (%)</Label>
                    <Input
                      type="number"
                      value={criteria.revenueGrowthPct}
                      onChange={(e) => setCriteria({...criteria, revenueGrowthPct: parseFloat(e.target.value) || 0})}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Min Earnings Growth (%)</Label>
                    <Input
                      type="number"
                      value={criteria.earningsGrowthPct}
                      onChange={(e) => setCriteria({...criteria, earningsGrowthPct: parseFloat(e.target.value) || 0})}
                    />
                  </div>
                </TabsContent>

                <TabsContent value="technical" className="space-y-4">
                  {/* Price Range */}
                  <div className="space-y-2">
                    <Label>Price Range ($)</Label>
                    <div className="grid grid-cols-2 gap-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={criteria.priceMin}
                        onChange={(e) => setCriteria({...criteria, priceMin: parseFloat(e.target.value) || 0})}
                      />
                      <Input
                        type="number"
                        placeholder="Max"
                        value={criteria.priceMax}
                        onChange={(e) => setCriteria({...criteria, priceMax: parseFloat(e.target.value) || 1000})}
                      />
                    </div>
                  </div>

                  {/* Volume */}
                  <div className="space-y-2">
                    <Label>Min Volume</Label>
                    <Input
                      type="number"
                      value={criteria.volumeMin}
                      onChange={(e) => setCriteria({...criteria, volumeMin: parseFloat(e.target.value) || 0})}
                    />
                  </div>

                  {/* RSI */}
                  <div className="space-y-2">
                    <Label>RSI Range</Label>
                    <Slider
                      value={[criteria.rsiMin, criteria.rsiMax]}
                      onValueChange={(values) => setCriteria({...criteria, rsiMin: values[0], rsiMax: values[1]})}
                      max={100}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>{criteria.rsiMin}</span>
                      <span>{criteria.rsiMax}</span>
                    </div>
                  </div>

                  {/* Moving Averages */}
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={criteria.sma20Above}
                        onCheckedChange={(checked) => setCriteria({...criteria, sma20Above: checked})}
                      />
                      <Label>Price above 20-day SMA</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={criteria.sma50Above}
                        onCheckedChange={(checked) => setCriteria({...criteria, sma50Above: checked})}
                      />
                      <Label>Price above 50-day SMA</Label>
                    </div>
                  </div>

                  {/* Beta */}
                  <div className="space-y-2">
                    <Label>Beta Range</Label>
                    <Slider
                      value={[criteria.betaMin, criteria.betaMax]}
                      onValueChange={(values) => setCriteria({...criteria, betaMin: values[0], betaMax: values[1]})}
                      max={3}
                      step={0.1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>{criteria.betaMin}</span>
                      <span>{criteria.betaMax}</span>
                    </div>
                  </div>

                  {/* Volatility */}
                  <div className="space-y-2">
                    <Label>Max Volatility (%)</Label>
                    <Input
                      type="number"
                      value={criteria.volatilityMax}
                      onChange={(e) => setCriteria({...criteria, volatilityMax: parseFloat(e.target.value) || 50})}
                    />
                  </div>
                </TabsContent>

                <TabsContent value="other" className="space-y-4">
                  {/* Dividend Yield */}
                  <div className="space-y-2">
                    <Label>Dividend Yield (%)</Label>
                    <div className="grid grid-cols-2 gap-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={criteria.dividendYieldMin}
                        onChange={(e) => setCriteria({...criteria, dividendYieldMin: parseFloat(e.target.value) || 0})}
                      />
                      <Input
                        type="number"
                        placeholder="Max"
                        value={criteria.dividendYieldMax}
                        onChange={(e) => setCriteria({...criteria, dividendYieldMax: parseFloat(e.target.value) || 10})}
                      />
                    </div>
                  </div>

                  {/* Sectors */}
                  <div className="space-y-2">
                    <Label>Sectors</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select sectors" />
                      </SelectTrigger>
                      <SelectContent>
                        {sectors.map(sector => (
                          <SelectItem key={sector} value={sector}>{sector}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Exchanges */}
                  <div className="space-y-2">
                    <Label>Exchanges</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select exchanges" />
                      </SelectTrigger>
                      <SelectContent>
                        {exchanges.map(exchange => (
                          <SelectItem key={exchange} value={exchange}>{exchange}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </TabsContent>
              </Tabs>

              {/* Preset Screens */}
              <div className="mt-6 space-y-2">
                <Label>Preset Screens</Label>
                <div className="grid grid-cols-1 gap-2">
                  {Object.keys(presetScreens).map(preset => (
                    <Button
                      key={preset}
                      variant="outline"
                      size="sm"
                      onClick={() => applyPreset(preset as keyof typeof presetScreens)}
                      className="justify-start"
                    >
                      {preset}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2" />
                  Screening Results
                </div>
                <Badge variant="outline">{results.length} stocks found</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-64">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Screening stocks...</p>
                  </div>
                </div>
              ) : results.length > 0 ? (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('symbol')}
                        >
                          Symbol {sortBy === 'symbol' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead>Company</TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('price')}
                        >
                          Price {sortBy === 'price' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('marketCap')}
                        >
                          Market Cap {sortBy === 'marketCap' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('peRatio')}
                        >
                          P/E {sortBy === 'peRatio' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('roe')}
                        >
                          ROE {sortBy === 'roe' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('revenueGrowth')}
                        >
                          Rev Growth {sortBy === 'revenueGrowth' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('dividendYield')}
                        >
                          Div Yield {sortBy === 'dividendYield' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('beta')}
                        >
                          Beta {sortBy === 'beta' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                        <TableHead>Sector</TableHead>
                        <TableHead 
                          className="cursor-pointer hover:bg-gray-50"
                          onClick={() => sortResults('score')}
                        >
                          Score {sortBy === 'score' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.map((stock) => (
                        <TableRow key={stock.symbol} className="hover:bg-gray-50">
                          <TableCell className="font-medium">{stock.symbol}</TableCell>
                          <TableCell className="max-w-32 truncate">{stock.name}</TableCell>
                          <TableCell>${formatNumber(stock.price)}</TableCell>
                          <TableCell>{formatCurrency(stock.marketCap)}</TableCell>
                          <TableCell>{formatNumber(stock.peRatio, 1)}</TableCell>
                          <TableCell className={stock.roe > 15 ? 'text-green-600' : ''}>
                            {formatPercentage(stock.roe)}
                          </TableCell>
                          <TableCell className={stock.revenueGrowth > 0 ? 'text-green-600' : 'text-red-600'}>
                            {formatPercentage(stock.revenueGrowth)}
                          </TableCell>
                          <TableCell>{formatPercentage(stock.dividendYield)}</TableCell>
                          <TableCell>{formatNumber(stock.beta, 2)}</TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {stock.sector}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge className={getScoreColor(stock.score)}>
                              {formatNumber(stock.score, 0)}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
                  <p className="text-gray-600">
                    Try adjusting your screening criteria or run a screen to see results.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
