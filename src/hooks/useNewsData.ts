import { useState, useEffect } from 'react';

// Define the shape of a news item
interface NewsItem {
  id: string;
  title: string;
  source: string;
  time: string;
  summary: string;
  relevance: 'high' | 'medium' | 'low';
}

// Backend news item interface
interface BackendNewsItem {
  title: string;
  description: string;
  url: string;
  source: string;
  published_at: string;
  symbol: string | null;
  sentiment_score: number;
  sentiment_label: string;
  sentiment_confidence: number;
}

// Mock data as fallback
const mockNewsData: NewsItem[] = [
  {
    id: '1',
    title: 'Fed Signals Potential Rate Cuts Ahead',
    source: 'Reuters',
    time: '2h ago',
    summary: 'Federal Reserve officials hint at possible interest rate adjustments in upcoming meetings...',
    relevance: 'high'
  },
  {
    id: '2',
    title: 'Tech Earnings Beat Expectations',
    source: 'Bloomberg',
    time: '4h ago',
    summary: 'Major technology companies report stronger than expected quarterly results...',
    relevance: 'medium'
  },
  {
    id: '3',
    title: 'Oil Prices Rise on Supply Concerns',
    source: 'WSJ',
    time: '6h ago',
    summary: 'Crude oil futures climb as geopolitical tensions raise supply concerns...',
    relevance: 'medium'
  },
  {
    id: '4',
    title: 'Dollar Strengthens Against Euro',
    source: 'Financial Times',
    time: '8h ago',
    summary: 'US Dollar gains ground against major currencies amid economic data...',
    relevance: 'low'
  },
];

/**
 * Custom hook to fetch news data from the backend API.
 *
 * @returns An object containing the news data, loading state, and error state.
 */
export const useNewsData = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const convertBackendNewsToNewsItem = (backendNews: BackendNewsItem[]): NewsItem[] => {
    return backendNews.map((item, index) => {
      // Calculate time ago from published_at
      const publishedDate = new Date(item.published_at);
      const now = new Date();
      const diffMs = now.getTime() - publishedDate.getTime();
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      const diffDays = Math.floor(diffHours / 24);
      
      let timeAgo: string;
      if (diffDays > 0) {
        timeAgo = `${diffDays}d ago`;
      } else if (diffHours > 0) {
        timeAgo = `${diffHours}h ago`;
      } else {
        const diffMinutes = Math.floor(diffMs / (1000 * 60));
        timeAgo = `${diffMinutes}m ago`;
      }

      // Determine relevance based on sentiment confidence and score
      let relevance: 'high' | 'medium' | 'low';
      if (item.sentiment_confidence > 0.8 && Math.abs(item.sentiment_score) > 0.5) {
        relevance = 'high';
      } else if (item.sentiment_confidence > 0.6) {
        relevance = 'medium';
      } else {
        relevance = 'low';
      }

      return {
        id: `news-${index}`,
        title: item.title,
        source: item.source,
        time: timeAgo,
        summary: item.description || 'No description available',
        relevance
      };
    });
  };

  useEffect(() => {
    const fetchNews = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch('/api/market-data/news');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const backendNews: BackendNewsItem[] = await response.json();
        
        if (Array.isArray(backendNews) && backendNews.length > 0) {
          const convertedNews = convertBackendNewsToNewsItem(backendNews);
          setNews(convertedNews.slice(0, 20)); // Limit to 20 most recent items
        } else {
          // Fallback to mock data if no real news available
          setNews(mockNewsData);
        }
        
      } catch (err) {
        console.error('Failed to fetch news:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch news data');
        
        // Fallback to mock data on error
        setNews(mockNewsData);
      } finally {
        setIsLoading(false);
      }
    };

    fetchNews();
    
    // Refresh news every 5 minutes
    const interval = setInterval(fetchNews, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, []); // Empty dependency array ensures this runs only once on mount

  return { news, isLoading, error };
};
