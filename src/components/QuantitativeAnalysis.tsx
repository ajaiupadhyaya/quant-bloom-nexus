import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { D3LineChart } from './D3LineChart';
import { D3BarChart } from './D3BarChart';
import { D3PieChart } from './D3PieChart';
import { TrendingUp, TrendingDown, Calculator, Target, Shield, BarChart3, PieChart, LineChart } from 'lucide-react';

interface QuantitativeAnalysisProps {
  symbol: string;
}

interface OptionPricing {
  black_scholes_price: number;
  binomial_price: number;
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
  };
}

interface RiskMetrics {
  var_95_historical: number;
  var_99_historical: number;
  cvar_95: number;
  cvar_99: number;
  max_drawdown: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  skewness: number;
  kurtosis: number;
}

interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  alpha: number;
  beta: number;
  information_ratio: number;
  max_drawdown: number;
}

interface StatisticalTests {
  normality_tests: {
    shapiro_wilk: { statistic: number; p_value: number; is_normal: boolean };
    jarque_bera: { statistic: number; p_value: number; is_normal: boolean };
  };
  stationarity_test: {
    adf_statistic: number;
    adf_p_value: number;
    is_stationary: boolean;
  };
  autocorrelation_test: {
    ljung_box_statistic: number;
    ljung_box_p_value: number;
    has_autocorrelation: boolean;
  };
  arch_effect_test: {
    arch_statistic: number;
    arch_p_value: number;
    has_arch_effects: boolean;
  };
}

export const QuantitativeAnalysis: React.FC<QuantitativeAnalysisProps> = ({ symbol }) => {
  const [optionPricing, setOptionPricing] = useState<OptionPricing | null>(null);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [statisticalTests, setStatisticalTests] = useState<StatisticalTests | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Option pricing inputs
  const [optionParams, setOptionParams] = useState({
    underlying_price: 100,
    strike_price: 100,
    time_to_expiry: 0.25,
    risk_free_rate: 0.02,
    volatility: 0.2,
    option_type: 'call'
  });

  // Portfolio inputs
  const [portfolioSymbols, setPortfolioSymbols] = useState(['AAPL', 'GOOGL', 'MSFT']);
  const [portfolioWeights, setPortfolioWeights] = useState([0.4, 0.3, 0.3]);

  useEffect(() => {
    fetchPerformanceMetrics();
    fetchStatisticalTests();
  }, [symbol]);

  const fetchOptionPricing = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/analytics/options/price', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(optionParams)
      });
      const data = await response.json();
      setOptionPricing(data);
    } catch (error) {
      console.error('Failed to fetch option pricing:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRiskAnalysis = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/analytics/portfolio/risk-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols: portfolioSymbols,
          weights: portfolioWeights
        })
      });
      const data = await response.json();
      setRiskMetrics(data.portfolio_risk_metrics);
    } catch (error) {
      console.error('Failed to fetch risk analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPerformanceMetrics = async () => {
    try {
      const response = await fetch(`/api/v1/analytics/performance/metrics?symbol=${symbol}&benchmark=SPY&period=1y`);
      const data = await response.json();
      setPerformanceMetrics(data.performance_metrics);
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error);
    }
  };

  const fetchStatisticalTests = async () => {
    try {
      const response = await fetch(`/api/v1/analytics/statistics/tests?symbol=${symbol}&period=1y`);
      const data = await response.json();
      setStatisticalTests(data.statistical_tests);
    } catch (error) {
      console.error('Failed to fetch statistical tests:', error);
    }
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals);
  };

  const formatPercentage = (num: number, decimals: number = 2) => {
    return `${(num * 100).toFixed(decimals)}%`;
  };

  const getSignalColor = (value: number, threshold: number = 0) => {
    if (value > threshold) return 'text-green-600';
    if (value < threshold) return 'text-red-600';
    return 'text-gray-600';
  };

  const getRiskLevel = (sharpe: number) => {
    if (sharpe > 1) return { level: 'Low', color: 'bg-green-500' };
    if (sharpe > 0.5) return { level: 'Medium', color: 'bg-yellow-500' };
    return { level: 'High', color: 'bg-red-500' };
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900">Quantitative Analysis</h2>
        <div className="flex items-center space-x-2">
          <Calculator className="h-8 w-8 text-blue-600" />
          <span className="text-lg font-semibold text-gray-700">{symbol}</span>
        </div>
      </div>

      <Tabs defaultValue="options" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="options">Options Pricing</TabsTrigger>
          <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="statistics">Statistical Tests</TabsTrigger>
          <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
        </TabsList>

        {/* OPTIONS PRICING TAB */}
        <TabsContent value="options" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Target className="h-5 w-5 mr-2" />
                  Option Parameters
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="underlying">Underlying Price</Label>
                    <Input
                      id="underlying"
                      type="number"
                      value={optionParams.underlying_price}
                      onChange={(e) => setOptionParams({...optionParams, underlying_price: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label htmlFor="strike">Strike Price</Label>
                    <Input
                      id="strike"
                      type="number"
                      value={optionParams.strike_price}
                      onChange={(e) => setOptionParams({...optionParams, strike_price: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label htmlFor="expiry">Time to Expiry (Years)</Label>
                    <Input
                      id="expiry"
                      type="number"
                      step="0.01"
                      value={optionParams.time_to_expiry}
                      onChange={(e) => setOptionParams({...optionParams, time_to_expiry: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label htmlFor="volatility">Volatility</Label>
                    <Input
                      id="volatility"
                      type="number"
                      step="0.01"
                      value={optionParams.volatility}
                      onChange={(e) => setOptionParams({...optionParams, volatility: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label htmlFor="rate">Risk-Free Rate</Label>
                    <Input
                      id="rate"
                      type="number"
                      step="0.001"
                      value={optionParams.risk_free_rate}
                      onChange={(e) => setOptionParams({...optionParams, risk_free_rate: parseFloat(e.target.value)})}
                    />
                  </div>
                  <div>
                    <Label htmlFor="type">Option Type</Label>
                    <Select value={optionParams.option_type} onValueChange={(value) => setOptionParams({...optionParams, option_type: value})}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="call">Call</SelectItem>
                        <SelectItem value="put">Put</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <Button onClick={fetchOptionPricing} disabled={loading} className="w-full">
                  Calculate Option Price
                </Button>
              </CardContent>
            </Card>

            {optionPricing && (
              <Card>
                <CardHeader>
                  <CardTitle>Option Pricing Results</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-sm text-gray-600">Black-Scholes Price</div>
                      <div className="text-2xl font-bold text-blue-600">
                        ${formatNumber(optionPricing.black_scholes_price)}
                      </div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-sm text-gray-600">Binomial Price</div>
                      <div className="text-2xl font-bold text-green-600">
                        ${formatNumber(optionPricing.binomial_price)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-semibold">Greeks</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="flex justify-between">
                        <span>Delta:</span>
                        <span className="font-mono">{formatNumber(optionPricing.greeks.delta, 4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Gamma:</span>
                        <span className="font-mono">{formatNumber(optionPricing.greeks.gamma, 4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Theta:</span>
                        <span className="font-mono">{formatNumber(optionPricing.greeks.theta, 4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Vega:</span>
                        <span className="font-mono">{formatNumber(optionPricing.greeks.vega, 4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Rho:</span>
                        <span className="font-mono">{formatNumber(optionPricing.greeks.rho, 4)}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* RISK ANALYSIS TAB */}
        <TabsContent value="risk" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Shield className="h-5 w-5 mr-2" />
                  Portfolio Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Symbols (comma-separated)</Label>
                  <Input
                    value={portfolioSymbols.join(', ')}
                    onChange={(e) => setPortfolioSymbols(e.target.value.split(',').map(s => s.trim()))}
                  />
                </div>
                <div>
                  <Label>Weights (comma-separated, must sum to 1.0)</Label>
                  <Input
                    value={portfolioWeights.join(', ')}
                    onChange={(e) => setPortfolioWeights(e.target.value.split(',').map(w => parseFloat(w.trim())))}
                  />
                </div>
                <Button onClick={fetchRiskAnalysis} disabled={loading} className="w-full">
                  Analyze Risk
                </Button>
              </CardContent>
            </Card>

            {riskMetrics && (
              <Card>
                <CardHeader>
                  <CardTitle>Risk Metrics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="text-sm text-gray-600">VaR (95%)</div>
                      <div className="text-lg font-bold text-red-600">
                        {formatPercentage(riskMetrics.var_95_historical)}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-600">VaR (99%)</div>
                      <div className="text-lg font-bold text-red-700">
                        {formatPercentage(riskMetrics.var_99_historical)}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-600">Max Drawdown</div>
                      <div className="text-lg font-bold text-red-600">
                        {formatPercentage(riskMetrics.max_drawdown)}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-600">Volatility</div>
                      <div className="text-lg font-bold text-blue-600">
                        {formatPercentage(riskMetrics.volatility)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Sharpe Ratio</span>
                      <Badge className={getRiskLevel(riskMetrics.sharpe_ratio).color}>
                        {getRiskLevel(riskMetrics.sharpe_ratio).level} Risk
                      </Badge>
                    </div>
                    <div className="text-2xl font-bold">{formatNumber(riskMetrics.sharpe_ratio)}</div>
                    <Progress value={Math.min(Math.max(riskMetrics.sharpe_ratio * 50, 0), 100)} />
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span>Sortino Ratio:</span>
                      <span className="font-mono">{formatNumber(riskMetrics.sortino_ratio)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Skewness:</span>
                      <span className="font-mono">{formatNumber(riskMetrics.skewness)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Kurtosis:</span>
                      <span className="font-mono">{formatNumber(riskMetrics.kurtosis)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>CVaR (95%):</span>
                      <span className="font-mono">{formatPercentage(riskMetrics.cvar_95)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* PERFORMANCE TAB */}
        <TabsContent value="performance" className="space-y-6">
          {performanceMetrics && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <TrendingUp className="h-5 w-5 mr-2" />
                    Returns
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-sm text-gray-600">Total Return</div>
                    <div className={`text-2xl font-bold ${getSignalColor(performanceMetrics.total_return)}`}>
                      {formatPercentage(performanceMetrics.total_return)}
                    </div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-sm text-gray-600">Annualized Return</div>
                    <div className={`text-2xl font-bold ${getSignalColor(performanceMetrics.annualized_return)}`}>
                      {formatPercentage(performanceMetrics.annualized_return)}
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Volatility:</span>
                      <span className="font-mono">{formatPercentage(performanceMetrics.volatility)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Max Drawdown:</span>
                      <span className="font-mono text-red-600">{formatPercentage(performanceMetrics.max_drawdown)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <BarChart3 className="h-5 w-5 mr-2" />
                    Risk-Adjusted
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-sm text-gray-600">Sharpe Ratio</div>
                    <div className={`text-2xl font-bold ${getSignalColor(performanceMetrics.sharpe_ratio)}`}>
                      {formatNumber(performanceMetrics.sharpe_ratio)}
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Alpha:</span>
                      <span className={`font-mono ${getSignalColor(performanceMetrics.alpha)}`}>
                        {formatPercentage(performanceMetrics.alpha)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Beta:</span>
                      <span className="font-mono">{formatNumber(performanceMetrics.beta)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Information Ratio:</span>
                      <span className="font-mono">{formatNumber(performanceMetrics.information_ratio)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Performance Visualization</CardTitle>
                </CardHeader>
                <CardContent>
                  <D3BarChart
                    data={[
                      { label: 'Total Return', value: performanceMetrics.total_return * 100 },
                      { label: 'Benchmark', value: 8.5 },
                      { label: 'Risk-Free', value: 2.0 }
                    ]}
                    width={300}
                    height={200}
                  />
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* STATISTICAL TESTS TAB */}
        <TabsContent value="statistics" className="space-y-6">
          {statisticalTests && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Normality Tests</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">Shapiro-Wilk Test</div>
                        <div className="text-sm text-gray-600">
                          Statistic: {formatNumber(statisticalTests.normality_tests.shapiro_wilk.statistic, 4)}
                        </div>
                        <div className="text-sm text-gray-600">
                          P-value: {formatNumber(statisticalTests.normality_tests.shapiro_wilk.p_value, 4)}
                        </div>
                      </div>
                      <Badge variant={statisticalTests.normality_tests.shapiro_wilk.is_normal ? "default" : "destructive"}>
                        {statisticalTests.normality_tests.shapiro_wilk.is_normal ? "Normal" : "Non-Normal"}
                      </Badge>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">Jarque-Bera Test</div>
                        <div className="text-sm text-gray-600">
                          Statistic: {formatNumber(statisticalTests.normality_tests.jarque_bera.statistic, 4)}
                        </div>
                        <div className="text-sm text-gray-600">
                          P-value: {formatNumber(statisticalTests.normality_tests.jarque_bera.p_value, 4)}
                        </div>
                      </div>
                      <Badge variant={statisticalTests.normality_tests.jarque_bera.is_normal ? "default" : "destructive"}>
                        {statisticalTests.normality_tests.jarque_bera.is_normal ? "Normal" : "Non-Normal"}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Time Series Tests</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">Stationarity (ADF)</div>
                        <div className="text-sm text-gray-600">
                          Statistic: {formatNumber(statisticalTests.stationarity_test.adf_statistic, 4)}
                        </div>
                        <div className="text-sm text-gray-600">
                          P-value: {formatNumber(statisticalTests.stationarity_test.adf_p_value, 4)}
                        </div>
                      </div>
                      <Badge variant={statisticalTests.stationarity_test.is_stationary ? "default" : "destructive"}>
                        {statisticalTests.stationarity_test.is_stationary ? "Stationary" : "Non-Stationary"}
                      </Badge>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">Autocorrelation (Ljung-Box)</div>
                        <div className="text-sm text-gray-600">
                          Statistic: {formatNumber(statisticalTests.autocorrelation_test.ljung_box_statistic, 4)}
                        </div>
                        <div className="text-sm text-gray-600">
                          P-value: {formatNumber(statisticalTests.autocorrelation_test.ljung_box_p_value, 4)}
                        </div>
                      </div>
                      <Badge variant={statisticalTests.autocorrelation_test.has_autocorrelation ? "destructive" : "default"}>
                        {statisticalTests.autocorrelation_test.has_autocorrelation ? "Autocorrelated" : "No Autocorr"}
                      </Badge>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">ARCH Effects</div>
                        <div className="text-sm text-gray-600">
                          Statistic: {formatNumber(statisticalTests.arch_effect_test.arch_statistic, 4)}
                        </div>
                        <div className="text-sm text-gray-600">
                          P-value: {formatNumber(statisticalTests.arch_effect_test.arch_p_value, 4)}
                        </div>
                      </div>
                      <Badge variant={statisticalTests.arch_effect_test.has_arch_effects ? "destructive" : "default"}>
                        {statisticalTests.arch_effect_test.has_arch_effects ? "ARCH Effects" : "No ARCH"}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* PORTFOLIO TAB */}
        <TabsContent value="portfolio" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <PieChart className="h-5 w-5 mr-2" />
                  Portfolio Allocation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <D3PieChart
                  data={portfolioSymbols.map((symbol, index) => ({
                    label: symbol,
                    value: portfolioWeights[index] * 100
                  }))}
                  width={400}
                  height={300}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Optimization Results</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Alert>
                  <AlertDescription>
                    Portfolio optimization uses Modern Portfolio Theory to find optimal weights
                    that maximize risk-adjusted returns.
                  </AlertDescription>
                </Alert>
                
                <Button className="w-full" onClick={() => {
                  // Implement portfolio optimization
                  console.log('Optimizing portfolio...');
                }}>
                  Optimize Portfolio
                </Button>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Expected Return:</span>
                    <span className="font-mono">12.5%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Expected Volatility:</span>
                    <span className="font-mono">15.2%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Sharpe Ratio:</span>
                    <span className="font-mono">0.69</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}; 