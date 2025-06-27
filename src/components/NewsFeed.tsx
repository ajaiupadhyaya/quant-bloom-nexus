
import React from 'react';
import { Clock, ExternalLink, AlertTriangle } from 'lucide-react';
import { useNewsData } from '@/hooks/useNewsData';
import { Skeleton } from '@/components/ui/skeleton';

// Define the shape of a single news item, which is now managed by the hook
interface NewsItem {
  id: string;
  title: string;
  source: string;
  time: string;
  summary: string;
  relevance: 'high' | 'medium' | 'low';
}

// A functional component to render a skeleton loader for a news item
const NewsItemSkeleton = () => (
  <div className="border-b border-terminal-border/30 pb-3 last:border-b-0">
    <div className="flex items-start justify-between mb-2">
      <Skeleton className="w-2 h-2 rounded-full mt-1" />
      <Skeleton className="w-16 h-4" />
    </div>
    <Skeleton className="w-3/4 h-5 mb-1" />
    <Skeleton className="w-full h-8 mb-2" />
    <div className="flex items-center justify-between">
      <Skeleton className="w-24 h-4" />
      <Skeleton className="w-4 h-4" />
    </div>
  </div>
);

export const NewsFeed = () => {
  const { news, isLoading, error } = useNewsData();

  const getRelevanceColor = (relevance: string) => {
    switch (relevance) {
      case 'high': return 'bg-terminal-red';
      case 'medium': return 'bg-terminal-amber';
      case 'low': return 'bg-terminal-cyan';
      default: return 'bg-terminal-muted';
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <>
          {[...Array(4)].map((_, index) => (
            <NewsItemSkeleton key={index} />
          ))}
        </>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-terminal-red">
          <AlertTriangle className="w-8 h-8 mb-2" />
          <p className="text-sm font-semibold">Error Loading News</p>
          <p className="text-xs text-center mt-1">{error}</p>
        </div>
      );
    }

    if (news.length === 0) {
        return (
            <div className="flex items-center justify-center h-full text-terminal-muted">
                <p>No news available at the moment.</p>
            </div>
        );
    }

    return news.map((item: NewsItem) => (
      <div key={item.id} className="border-b border-terminal-border/30 pb-3 last:border-b-0">
        <div className="flex items-start justify-between mb-2">
          <div className={`w-2 h-2 rounded-full mt-1 ${getRelevanceColor(item.relevance)}`} />
          <div className="flex items-center text-xs text-terminal-muted">
            <Clock className="w-3 h-3 mr-1" />
            {item.time}
          </div>
        </div>
        <h3 className="text-sm font-medium text-terminal-text mb-1 hover:text-terminal-cyan cursor-pointer">
          {item.title}
        </h3>
        <p className="text-xs text-terminal-muted mb-2 line-clamp-2">
          {item.summary}
        </p>
        <div className="flex items-center justify-between">
          <span className="text-xs text-terminal-orange font-medium">
            {item.source}
          </span>
          <ExternalLink className="w-3 h-3 text-terminal-muted hover:text-terminal-cyan cursor-pointer" />
        </div>
      </div>
    ));
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <h2 className="text-sm font-semibold text-terminal-orange">NEWS FEED</h2>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {renderContent()}
      </div>
    </div>
  );
};
