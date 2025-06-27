
import React, { useRef, useEffect, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickSeriesPartialOptions, LineSeriesPartialOptions, HistogramSeriesPartialOptions } from 'lightweight-charts';
import { useHistoricalData } from '@/hooks/useHistoricalData';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle, TrendingUp, BarChart } from 'lucide-react';

interface AdvancedPriceChartProps {
  symbol: string;
}

// Mock WebSocket for real-time data (replace with actual WebSocket logic)
const mockWebSocket = {
  onmessage: (event: MessageEvent) => {},
  send: (message: string) => console.log('WS Send:', message),
  close: () => console.log('WS Closed'),
  readyState: 1, // Simulates open connection
  OPEN: 1,
};

export const AdvancedPriceChart: React.FC<AdvancedPriceChartProps> = ({ symbol }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const ma50SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ma200SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  const { data: historicalData, isLoading, error } = useHistoricalData(symbol);
  const [timeframe, setTimeframe] = useState('1D'); // Default timeframe

  // Function to add technical indicators (placeholder)
  const addTechnicalIndicator = useCallback((indicatorName: string) => {
    console.log(`Adding technical indicator: ${indicatorName} for ${symbol}`);
    // In a real application, this would trigger a backend call
    // to calculate the indicator and then add a new series to the chart.
    // Example: fetch(`/api/indicators/${symbol}?indicator=${indicatorName}&timeframe=${timeframe}`)
  }, [symbol, timeframe]);

  useEffect(() => {
    if (isLoading || error || !chartContainerRef.current) return;

    if (!chartRef.current) {
      // Initialize chart
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight,
        layout: {
          background: { color: '#1a1a1a' },
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: '#333333' },
          horzLines: { color: '#333333' },
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          borderVisible: false,
        },
        rightPriceScale: {
          borderVisible: false,
        },
      });
      chartRef.current = chart;

      // Add Candlestick Series
      candlestickSeriesRef.current = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      } as CandlestickSeriesPartialOptions);

      // Add Volume Series
      volumeSeriesRef.current = chart.addHistogramSeries({
        color: '#2962FF',
        priceFormat: {
          type: 'volume',
        },
        overlay: true,
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      } as HistogramSeriesPartialOptions);

      // Add Moving Average Series
      ma50SeriesRef.current = chart.addLineSeries({
        color: '#F59E0B', // Amber
        lineWidth: 1,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
      } as LineSeriesPartialOptions);

      ma200SeriesRef.current = chart.addLineSeries({
        color: '#EF4444', // Red
        lineWidth: 1,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
      } as LineSeriesPartialOptions);

      // Handle resize
      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
          });
        }
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }
      };
    }

    // Update chart data when historicalData changes
    if (historicalData.length > 0) {
      const formattedCandlestickData = historicalData.map(d => ({
        time: new Date(d.date).getTime() / 1000, // Lightweight-charts expects Unix timestamp in seconds
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));

      const formattedVolumeData = historicalData.map(d => ({
        time: new Date(d.date).getTime() / 1000,
        value: d.volume,
        color: d.close > d.open ? 'rgba(38, 166, 154, 0.4)' : 'rgba(239, 83, 80, 0.4)',
      }));

      // Mock MA data (replace with actual calculated MA data)
      const formattedMa50Data = historicalData.map(d => ({
        time: new Date(d.date).getTime() / 1000,
        value: d.close * (1 + (Math.random() - 0.5) * 0.02), // Mock MA
      }));
      const formattedMa200Data = historicalData.map(d => ({
        time: new Date(d.date).getTime() / 1000,
        value: d.close * (1 + (Math.random() - 0.5) * 0.05), // Mock MA
      }));

      candlestickSeriesRef.current?.setData(formattedCandlestickData);
      volumeSeriesRef.current?.setData(formattedVolumeData);
      ma50SeriesRef.current?.setData(formattedMa50Data);
      ma200SeriesRef.current?.setData(formattedMa200Data);

      // Fit content to screen
      chartRef.current?.timeScale().fitContent();
    }
  }, [historicalData, isLoading, error]);

  // Real-time data updates (mocked WebSocket)
  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;

    // Simulate real-time updates
    const interval = setInterval(() => {
      const lastData = historicalData[historicalData.length - 1];
      if (!lastData) return;

      const newTime = new Date(new Date(lastData.date).getTime() + 24 * 60 * 60 * 1000).getTime() / 1000; // Next day
      const newOpen = lastData.close;
      const newClose = newOpen * (1 + (Math.random() - 0.5) * 0.01);
      const newHigh = Math.max(newOpen, newClose) * (1 + Math.random() * 0.005);
      const newLow = Math.min(newOpen, newClose) * (1 - Math.random() * 0.005);
      const newVolume = lastData.volume + Math.floor(Math.random() * 100000);

      const newCandle = {
        time: newTime,
        open: newOpen,
        high: newHigh,
        low: newLow,
        close: newClose,
      };

      const newVolumeBar = {
        time: newTime,
        value: newVolume,
        color: newClose > newOpen ? 'rgba(38, 166, 154, 0.4)' : 'rgba(239, 83, 80, 0.4)',
      };

      candlestickSeriesRef.current?.update(newCandle);
      volumeSeriesRef.current?.update(newVolumeBar);

      // Update mock MA data
      ma50SeriesRef.current?.update({ time: newTime, value: newClose * (1 + (Math.random() - 0.5) * 0.02) });
      ma200SeriesRef.current?.update({ time: newTime, value: newClose * (1 + (Math.random() - 0.5) * 0.05) });

    }, 5000); // Update every 5 seconds

    // In a real scenario, you'd connect to a WebSocket here:
    // const ws = new WebSocket(`ws://your-websocket-api/market-data/${symbol}`);
    // ws.onmessage = (event) => {
    //   const data = JSON.parse(event.data);
    //   // Process and update chart series
    // };
    // return () => ws.close();

    return () => clearInterval(interval);
  }, [historicalData, symbol]);

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

    return (
      <div className="h-full flex flex-col">
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
          
          <div className="flex space-x-2">
            {['1D', '1W', '1M', '3M', '1Y'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                  timeframe === tf 
                    ? 'bg-terminal-orange text-terminal-bg' 
                    : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-border'
                }`}
              >
                {tf}
              </button>
            ))}
            <button 
              onClick={() => addTechnicalIndicator('SMA')}
              className="px-3 py-1 text-xs font-medium rounded transition-colors text-terminal-muted hover:text-terminal-text hover:bg-terminal-border"
            >
              Add SMA
            </button>
          </div>
        </div>

        {/* Chart Container */}
        <div ref={chartContainerRef} className="flex-1" />
      </div>
    );
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      {renderChartContent()}
    </div>
  );
};
