import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { D3HeatmapChart } from './D3HeatmapChart';
import { D3NetworkGraph } from './D3NetworkGraph';
import { D3AdvancedLineChart } from './D3AdvancedLineChart';
import { TrendingUp, TrendingDown, BarChart3, Activity, Globe } from 'lucide-react';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  sector: string;
}

interface SectorData {
  sector: string;
  performance: number;
  volume: number;
  count: number;
}

export const EnhancedMarketOverview: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [sectorData, setSectorData] = useState<SectorData[]>([]);
  const [indexData, setIndexData] = useState<{ time: string; spx: number; ndx: number; dji: number }[]>([]);

  useEffect(() => {
    // Generate comprehensive market data
    const sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Industrial', 'Materials', 'Utilities'];
    const symbols = [
      { symbol: 'AAPL', sector: 'Technology', price: 175.43, mc: 2800000000000 },
      { symbol: 'MSFT', sector: 'Technology', price: 338.11, mc: 2500000000000 },
      { symbol: 'GOOGL', sector: 'Technology', price: 142.56, mc: 1800000000000 },
      { symbol: 'TSLA', sector: 'Technology', price: 248.87, mc: 800000000000 },
      { symbol: 'NVDA', sector: 'Technology', price: 875.28, mc: 2200000000000 },
      { symbol: 'META', sector: 'Technology', price: 327.89, mc: 900000000000 },
      { symbol: 'AMZN', sector: 'Consumer', price: 155.89, mc: 1600000000000 },
      { symbol: 'JPM', sector: 'Financial', price: 158.93, mc: 450000000000 },
      { symbol: 'JNJ', sector: 'Healthcare', price: 159.91, mc: 420000000000 },
      { symbol: 'XOM', sector: 'Energy', price: 118.77, mc: 480000000000 },
    ];

    const generatedMarketData = symbols.map(stock => ({
      ...stock,
      change: (Math.random() - 0.5) * 10,
      changePercent: (Math.random() - 0.5) * 5,
      volume: Math.floor(Math.random() * 50000000) + 5000000,
      marketCap: stock.mc
    }));

    const generatedSectorData = sectors.map(sector => ({
      sector,
      performance: (Math.random() - 0.5) * 4,
      volume: Math.floor(Math.random() * 1000000000) + 100000000,
      count: Math.floor(Math.random() * 50) + 10
    }));

    // Generate index time series data
    const now = new Date();
    const indexTimeSeries = Array.from({ length: 50 }, (_, i) => {
      const time = new Date(now.getTime() - (50 - i) * 300000);
      const baseSpx = 4521.23;
      const baseNdx = 15245.67;
      const baseDji = 34567.89;
      
      return {
        time: time.toISOString(),
        spx: baseSpx + Math.sin(i / 10) * 50 + Math.random() * 20,
        ndx: baseNdx + Math.sin(i / 8) * 200 + Math.random() * 100,
        dji: baseDji + Math.sin(i / 12) * 300 + Math.random() * 150
      };
    });

    setMarketData(generatedMarketData);
    setSectorData(generatedSectorData);
    setIndexData(indexTimeSeries);
  }, []);

  // Prepare heatmap data
  const heatmapData = marketData.map(stock => ({
    x: stock.sector,
    y: stock.symbol,
    value: stock.changePercent,
    label: `${stock.symbol}: ${stock.changePercent.toFixed(2)}%`
  }));

  // Prepare network graph data for sector correlations
  const networkNodes = sectorData.map((sector, index) => ({
    id: sector.sector,
    group: Math.floor(index / 3),
    size: sector.performance > 0 ? Math.abs(sector.performance) * 5 : 2,
    label: sector.sector
  }));

  const networkLinks = sectorData.flatMap((sector, i) => 
    sectorData.slice(i + 1).map(otherSector => ({
      source: sector.sector,
      target: otherSector.sector,
      value: Math.abs(sector.performance - otherSector.performance),
      type: 'correlation'
    }))
  ).slice(0, 15); // Limit connections for clarity

  // Prepare index comparison data
  const indexChartData = indexData.map(d => ({
    x: d.time,
    y: (d.spx / 4521.23) * 100, // Normalize to percentage
    volume: Math.random() * 1000000
  }));

  const topGainers = marketData
    .filter(stock => stock.changePercent > 0)
    .sort((a, b) => b.changePercent - a.changePercent)
    .slice(0, 5);

  const topLosers = marketData
    .filter(stock => stock.changePercent < 0)
    .sort((a, b) => a.changePercent - b.changePercent)
    .slice(0, 5);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <Globe className="w-8 h-8 text-orange-400" />
            <h1 className="text-3xl font-bold text-orange-400">Global Market Overview</h1>
          </div>
          <div className="flex items-center space-x-4">
            <Badge variant="outline" className="bg-green-900/50 text-green-400 border-green-400">
              Markets Open
            </Badge>
            <Badge variant="outline" className="bg-blue-900/50 text-blue-400 border-blue-400">
              Real-time Data
            </Badge>
          </div>
        </div>

        {/* Key Market Indices */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card className="bg-gray-900/50 border-orange-500/30">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">S&P 500</p>
                  <p className="text-2xl font-bold text-white">4,521.23</p>
                  <div className="flex items-center space-x-1">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-green-400">+12.45 (0.28%)</span>
                  </div>
                </div>
                <BarChart3 className="w-8 h-8 text-orange-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-orange-500/30">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">NASDAQ</p>
                  <p className="text-2xl font-bold text-white">15,245.67</p>
                  <div className="flex items-center space-x-1">
                    <TrendingDown className="w-4 h-4 text-red-400" />
                    <span className="text-red-400">-8.23 (-0.05%)</span>
                  </div>
                </div>
                <Activity className="w-8 h-8 text-cyan-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-orange-500/30">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">Dow Jones</p>
                  <p className="text-2xl font-bold text-white">34,567.89</p>
                  <div className="flex items-center space-x-1">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-green-400">+156.78 (0.45%)</span>
                  </div>
                </div>
                <BarChart3 className="w-8 h-8 text-green-400" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Advanced Visualizations */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-8">
        {/* Market Heatmap */}
        <Card className="bg-gray-900/50 border-orange-500/30">
          <CardHeader>
            <CardTitle className="text-orange-400">Sector Performance Heatmap</CardTitle>
          </CardHeader>
          <CardContent>
            <D3HeatmapChart
              data={heatmapData}
              width={600}
              height={400}
              title="Stock Performance by Sector"
              colorScheme={['#ff4757', '#ff6b35', '#ffa500', '#00d4ff', '#00ff88']}
            />
          </CardContent>
        </Card>

        {/* Sector Correlation Network */}
        <Card className="bg-gray-900/50 border-orange-500/30">
          <CardHeader>
            <CardTitle className="text-orange-400">Sector Correlation Network</CardTitle>
          </CardHeader>
          <CardContent>
            <D3NetworkGraph
              nodes={networkNodes}
              links={networkLinks}
              width={600}
              height={400}
              title="Sector Interconnections"
              colors={['#ff6b35', '#00d4ff', '#00ff88', '#ff4757', '#ffa500']}
            />
          </CardContent>
        </Card>
      </div>

      {/* Index Performance Chart */}
      <Card className="bg-gray-900/50 border-orange-500/30 mb-8">
        <CardHeader>
          <CardTitle className="text-orange-400">Market Index Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <D3AdvancedLineChart
            data={indexChartData}
            width={1200}
            height={400}
            colors={['#ff6b35', '#00d4ff', '#00ff88']}
            title="S&P 500 Intraday Performance"
            xLabel="Time"
            yLabel="Index Value"
            showVolume={true}
          />
        </CardContent>
      </Card>

      {/* Market Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Top Gainers */}
        <Card className="bg-gray-900/50 border-green-500/30">
          <CardHeader>
            <CardTitle className="text-green-400 flex items-center space-x-2">
              <TrendingUp className="w-5 h-5" />
              <span>Top Gainers</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {topGainers.map((stock, index) => (
                <div key={stock.symbol} className="flex items-center justify-between p-3 bg-green-900/20 rounded">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm text-gray-400">#{index + 1}</span>
                    <div>
                      <p className="font-semibold text-white">{stock.symbol}</p>
                      <p className="text-xs text-gray-400">{stock.sector}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-white">${stock.price.toFixed(2)}</p>
                    <p className="text-green-400">+{stock.changePercent.toFixed(2)}%</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Losers */}
        <Card className="bg-gray-900/50 border-red-500/30">
          <CardHeader>
            <CardTitle className="text-red-400 flex items-center space-x-2">
              <TrendingDown className="w-5 h-5" />
              <span>Top Losers</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {topLosers.map((stock, index) => (
                <div key={stock.symbol} className="flex items-center justify-between p-3 bg-red-900/20 rounded">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm text-gray-400">#{index + 1}</span>
                    <div>
                      <p className="font-semibold text-white">{stock.symbol}</p>
                      <p className="text-xs text-gray-400">{stock.sector}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-white">${stock.price.toFixed(2)}</p>
                    <p className="text-red-400">{stock.changePercent.toFixed(2)}%</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};