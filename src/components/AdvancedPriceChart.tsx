import React, { useState } from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle, TrendingUp, BarChart } from 'lucide-react';
import { D3CandlestickChart } from './D3CandlestickChart';
import { useHistoricalData } from '@/hooks/useHistoricalData';

interface AdvancedPriceChartProps {
  symbol: string;
}

export const AdvancedPriceChart: React.FC<AdvancedPriceChartProps> = ({ symbol }) => {
  const { data: historicalData, isLoading, error } = useHistoricalData(symbol);

  const renderChartContent = () => {
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

    if (!historicalData || historicalData.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-terminal-muted">
          <p>No historical data available for {symbol}.</p>
        </div>
      );
    }

    const latestData = historicalData[historicalData.length - 1];
    const priceChange = latestData.close - historicalData[0].close;
    const priceChangePercent = (priceChange / historicalData[0].close) * 100;

    // Format data for D3CandlestickChart
    const candlestickData = historicalData.map(d => ({
      date: d.date,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume
    }));

    return (
      <>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-terminal-border">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <BarChart className="h-5 w-5 text-terminal-orange" />
              <h2 className="text-lg font-semibold text-terminal-text">{symbol}</h2>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="text-right">
              <div className="text-lg font-bold text-terminal-text">
                ${latestData.close.toFixed(2)}
              </div>
              <div className={`text-sm ${priceChange >= 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%)
              </div>
            </div>
            
            <div className="text-xs text-terminal-muted">
              <div>Vol: {latestData.volume.toLocaleString()}</div>
              <div>H: ${latestData.high.toFixed(2)} L: ${latestData.low.toFixed(2)}</div>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="flex-1 p-4">
          <D3CandlestickChart
            data={candlestickData}
            width={800}
            height={400}
            title={`${symbol} - Advanced Price Chart`}
          />
        </div>
      </>
    );
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      {renderChartContent()}
    </div>
  );
};
