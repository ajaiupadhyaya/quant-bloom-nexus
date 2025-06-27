
import React from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { BarChart, TrendingUp, AlertTriangle } from 'lucide-react';
import { useHistoricalData } from '@/hooks/useHistoricalData';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

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
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <XAxis 
                            dataKey="date" 
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 11, fill: '#888888' }}
                        />
                        <YAxis 
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 11, fill: '#888888' }}
                            domain={['dataMin - 1', 'dataMax + 1']}
                        />
                        <Tooltip 
                            contentStyle={{
                                backgroundColor: '#1a1a1a',
                                border: '1px solid #333333',
                                borderRadius: '4px',
                                color: '#ffffff'
                            }}
                            formatter={(value: number) => [`${value.toFixed(2)}`, 'Price']}
                        />
                        <Line 
                            type="monotone" 
                            dataKey="close" 
                            stroke="#00d4ff" 
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, fill: '#00d4ff' }}
                        />
                    </LineChart>
                </ResponsiveContainer>
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
