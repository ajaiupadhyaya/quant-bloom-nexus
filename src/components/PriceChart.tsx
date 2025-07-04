import React from 'react';
import { BarChart, TrendingUp, AlertTriangle } from 'lucide-react';
import { useHistoricalData } from '@/hooks/useHistoricalData';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { D3AdvancedLineChart } from './D3AdvancedLineChart';

interface PriceChartProps {
  symbol: string;
}

export const PriceChart = ({ symbol }: PriceChartProps) => {
  const { data, isLoading, error } = useHistoricalData(symbol);

  const renderChart = () => {
    if (isLoading) {
      return <Skeleton className="w-full h-full" />;
    }

    if (error) {
      return (
        <Alert variant="destructive" className="h-full flex flex-col justify-center items-center">
          <AlertTriangle className="h-6 w-6" />
          <AlertTitle>Error Loading Chart</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      );
    }

    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-terminal-muted">
                <p>No data available for this symbol.</p>
            </div>
        );
    }

    const latestData = data[data.length - 1];
    const priceChange = latestData.close - data[0].close;

    // Format data for D3LineChart
    const d3Data = data.map(d => ({ x: d.date, y: d.close }));

    return (
        <>
            {/* Chart Header */}
            <div className="border-b border-terminal-border p-4 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                    <div>
                        <div className="flex items-center space-x-2">
                            <span className="font-mono font-bold text-lg text-terminal-cyan">{symbol}</span>
                            <BarChart className="w-4 h-4 text-terminal-muted" />
                        </div>
                        <div className="flex items-center space-x-4 mt-1">
                            <span className="financial-number text-xl font-semibold">
                                ${latestData.close.toFixed(2)}
                            </span>
                            <span className={`flex items-center text-sm ${
                                priceChange >= 0 ? 'status-positive' : 'status-negative'
                            }`}>
                                <TrendingUp className="w-3 h-3 mr-1" />
                                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({((priceChange / latestData.close) * 100).toFixed(2)}%)
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <div className="flex-1 p-4">
                <D3AdvancedLineChart
                  data={d3Data}
                  width={600}
                  height={320}
                  colors={['#ff6b35', '#00d4ff', '#00ff88']}
                  title={`${symbol} Price Analysis`}
                  xLabel="Date"
                  yLabel="Price ($)"
                  showVolume={false}
                />
            </div>
        </>
    );
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      {renderChart()}
    </div>
  );
};
