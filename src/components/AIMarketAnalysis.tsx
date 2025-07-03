import React, { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, Zap, Target, AlertTriangle, Loader2 } from 'lucide-react';
import { D3LineChart } from './D3LineChart';

interface PredictionData {
  timestamp: string;
  actual: number;
  predicted: number;
  confidence: number;
}

interface MarketSignal {
  id: string;
  type: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  reason: string;
  timeframe: string;
  accuracy: number;
}

interface AIAnalysisData {
  action: string;
  confidence: number;
  score: number;
  agent_consensus: {
    action: number;
    confidence: number;
    agent_votes: Array<{
      action: number;
      confidence: number;
      value_estimate: number;
      specialization: string;
    }>;
    consensus_strength: number;
  };
  transformer_prediction: {
    prediction: number;
    confidence: number;
    direction: number;
  };
}

export const AIMarketAnalysis = ({ symbol }: { symbol: string }) => {
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [signals, setSignals] = useState<MarketSignal[]>([]);
  const [modelAccuracy, setModelAccuracy] = useState(0.847);
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAIAnalysis = async () => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await fetch('/api/advanced-ai/comprehensive-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: symbol,
          timeframe: '1d',
          lookback_days: 30,
          include_regime_analysis: true,
          include_agent_consensus: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setAiAnalysis(data.data);
        setModelAccuracy(data.data.confidence);
        
        // Convert AI analysis to signals format
        const newSignals: MarketSignal[] = [];
        
        // Main AI signal
        const actionMap = { 'BUY': 'bullish', 'SELL': 'bearish', 'HOLD': 'neutral' };
        newSignals.push({
          id: 'main-ai',
          type: actionMap[data.data.action as keyof typeof actionMap] as 'bullish' | 'bearish' | 'neutral',
          strength: data.data.confidence,
          reason: `Advanced AI Engine recommends ${data.data.action} with ${(data.data.confidence * 100).toFixed(1)}% confidence`,
          timeframe: '1D',
          accuracy: data.data.confidence
        });

        // Transformer prediction signal
        if (data.data.transformer_prediction) {
          const direction = data.data.transformer_prediction.direction;
          newSignals.push({
            id: 'transformer',
            type: direction > 0 ? 'bullish' : direction < 0 ? 'bearish' : 'neutral',
            strength: data.data.transformer_prediction.confidence,
            reason: `Transformer model predicts ${direction > 0 ? 'upward' : direction < 0 ? 'downward' : 'sideways'} movement`,
            timeframe: '4H',
            accuracy: data.data.transformer_prediction.confidence
          });
        }

        // Agent consensus signals
        if (data.data.agent_consensus?.agent_votes) {
          data.data.agent_consensus.agent_votes.forEach((vote, index) => {
            const actionNames = ['HOLD', 'BUY', 'SELL'];
            const actionTypes = ['neutral', 'bullish', 'bearish'];
            
            newSignals.push({
              id: `agent-${index}`,
              type: actionTypes[vote.action] as 'bullish' | 'bearish' | 'neutral',
              strength: vote.confidence,
              reason: `${vote.specialization} agent suggests ${actionNames[vote.action]} (value: ${vote.value_estimate.toFixed(3)})`,
              timeframe: '1H',
              accuracy: vote.confidence
            });
          });
        }

        setSignals(newSignals);
        
        // Generate prediction chart data based on AI analysis
        const predictionData = [];
        const basePrice = 180; // This should come from current market price
        
        for (let i = 0; i < 24; i++) {
          const time = new Date(Date.now() + i * 3600000);
          const trend = data.data.transformer_prediction?.prediction || 0;
          const predicted = basePrice + (trend * i / 24 * 10) + Math.random() * 2;
          const actual = basePrice + Math.sin(i / 4) * 5 + Math.random() * 3;
          
          predictionData.push({
            timestamp: time.toLocaleTimeString(),
            actual: actual,
            predicted: predicted,
            confidence: data.data.confidence
          });
        }
        
        setPredictions(predictionData);
      } else {
        throw new Error(data.message || 'AI analysis failed');
      }
    } catch (error) {
      console.error('AI Analysis error:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch AI analysis');
      
      // Fallback to mock data on error
      generateMockData();
    } finally {
      setIsProcessing(false);
    }
  };

  const generateMockData = () => {
    // Original mock data generation as fallback
    const data = [];
    const basePrice = 180;
    
    for (let i = 0; i < 24; i++) {
      const time = new Date(Date.now() + i * 3600000);
      const actual = basePrice + Math.sin(i / 4) * 10 + Math.random() * 5;
      const predicted = actual + (Math.random() - 0.5) * 3;
      const confidence = 0.6 + Math.random() * 0.3;
      
      data.push({
        timestamp: time.toLocaleTimeString(),
        actual: actual,
        predicted: predicted,
        confidence: confidence
      });
    }
    setPredictions(data);

    const mockSignals = [
      {
        id: '1',
        type: 'bullish' as const,
        strength: 0.85,
        reason: 'LSTM Neural Network detected strong upward momentum pattern',
        timeframe: '4H',
        accuracy: 0.91
      },
      {
        id: '2',
        type: 'bearish' as const,
        strength: 0.62,
        reason: 'Transformer model identifies resistance level convergence',
        timeframe: '1D',
        accuracy: 0.78
      }
    ];
    setSignals(mockSignals);
  };

  useEffect(() => {
    fetchAIAnalysis();

    const interval = setInterval(() => {
      fetchAIAnalysis();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [symbol]);

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'bullish': return 'text-terminal-green';
      case 'bearish': return 'text-terminal-red';
      default: return 'text-terminal-cyan';
    }
  };

  const getSignalIcon = (type: string) => {
    switch (type) {
      case 'bullish': return TrendingUp;
      case 'bearish': return TrendingDown;
      default: return Target;
    }
  };

  // Format data for D3LineChart
  const predictionChartData = predictions.map((p, index) => ({
    x: index.toString(),
    y: p.predicted
  }));

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">AI MARKET ANALYSIS - {symbol}</h2>
          {isProcessing && <Loader2 className="w-4 h-4 text-terminal-cyan animate-spin" />}
        </div>
        <div className="text-xs text-terminal-muted">
          Model Accuracy: <span className="text-terminal-green">{(modelAccuracy * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      {error && (
        <div className="p-3 bg-terminal-red/20 border-b border-terminal-border">
          <div className="flex items-center space-x-2 text-terminal-red">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-xs">Error: {error}</span>
          </div>
        </div>
      )}
      
      <div className="flex-1 overflow-y-auto">
        {/* AI Analysis Summary */}
        {aiAnalysis && (
          <div className="p-3 border-b border-terminal-border/50">
            <h3 className="text-xs font-medium text-terminal-cyan mb-2">AI RECOMMENDATION</h3>
            <div className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
              <div className="flex items-center justify-between mb-2">
                <span className={`text-lg font-bold ${aiAnalysis.action === 'BUY' ? 'text-terminal-green' : aiAnalysis.action === 'SELL' ? 'text-terminal-red' : 'text-terminal-cyan'}`}>
                  {aiAnalysis.action}
                </span>
                <span className="text-xs text-terminal-muted">
                  Score: {aiAnalysis.score.toFixed(3)}
                </span>
              </div>
              <div className="text-xs text-terminal-muted">
                Confidence: <span className="text-terminal-cyan">{(aiAnalysis.confidence * 100).toFixed(1)}%</span>
                {aiAnalysis.agent_consensus && (
                  <span className="ml-4">
                    Consensus: <span className="text-terminal-cyan">{(aiAnalysis.agent_consensus.consensus_strength * 100).toFixed(0)}%</span>
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ML Predictions Chart */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-2">24H PRICE PREDICTION</h3>
          <div className="h-32">
            <D3LineChart
              data={predictionChartData}
              width={400}
              height={128}
              color="#00d4ff"
              title="24H Price Prediction"
              xLabel="Time"
              yLabel="Price"
            />
          </div>
        </div>

        {/* AI Signals */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">NEURAL NETWORK SIGNALS</h3>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {signals.map((signal) => {
              const IconComponent = getSignalIcon(signal.type);
              return (
                <div key={signal.id} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <IconComponent className={`w-4 h-4 ${getSignalColor(signal.type)}`} />
                      <span className={`text-sm font-medium ${getSignalColor(signal.type)} uppercase`}>
                        {signal.type}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-terminal-muted">{signal.timeframe}</span>
                      <div className={`w-2 h-2 rounded-full ${
                        signal.strength > 0.8 ? 'bg-terminal-green' : 
                        signal.strength > 0.6 ? 'bg-terminal-amber' : 'bg-terminal-red'
                      }`} />
                    </div>
                  </div>
                  
                  <p className="text-xs text-terminal-text mb-2">{signal.reason}</p>
                  
                  <div className="flex justify-between items-center">
                    <div className="text-xs text-terminal-muted">
                      Strength: <span className="text-terminal-cyan">{(signal.strength * 100).toFixed(0)}%</span>
                    </div>
                    <div className="text-xs text-terminal-muted">
                      Accuracy: <span className="text-terminal-green">{(signal.accuracy * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
