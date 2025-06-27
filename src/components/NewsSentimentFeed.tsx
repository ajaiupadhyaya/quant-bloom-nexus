
import React from 'react';
import { useNewsSentiment } from '@/hooks/useNewsSentiment';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle, Loader2 } from 'lucide-react';

interface NewsSentimentFeedProps {
  // No props needed for now, but can be extended later
}

export const NewsSentimentFeed: React.FC<NewsSentimentFeedProps> = () => {
  const { data, isLoading, error } = useNewsSentiment(10000); // Fetch every 10 seconds

  const getSentimentColorClass = (sentiment: 'positive' | 'negative' | 'neutral') => {
    switch (sentiment) {
      case 'positive': return 'bg-terminal-green';
      case 'negative': return 'bg-terminal-red';
      case 'neutral': return 'bg-terminal-muted';
      default: return 'bg-terminal-muted';
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-terminal-orange mb-4" />
          <p className="text-terminal-muted">Loading news sentiment...</p>
          <Skeleton className="w-full h-48 mt-4" />
        </div>
      );
    }

    if (error) {
      return (
        <Alert variant="destructive" className="h-full flex flex-col justify-center items-center">
          <AlertTriangle className="h-6 w-6" />
          <AlertTitle>Error Loading News Sentiment</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      );
    }

    if (!data || data.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-terminal-muted">
          <p>No news sentiment data available.</p>
        </div>
      );
    }

    return (
      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {data.map((item, index) => (
          <div key={index} className="flex items-start space-x-2 border-b border-terminal-border/30 pb-3 last:border-b-0">
            <div className={`w-2 h-2 rounded-full flex-shrink-0 mt-1 ${getSentimentColorClass(item.sentiment)}`} />
            <p className="text-xs text-terminal-text leading-tight">
              {item.headline}
            </p>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <h2 className="text-sm font-semibold text-terminal-orange">NEWS SENTIMENT</h2>
      </div>
      {renderContent()}
    </div>
  );
};
