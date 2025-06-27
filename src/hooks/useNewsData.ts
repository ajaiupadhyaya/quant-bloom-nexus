
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

// Mock data simulating a response from an API
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
 * Custom hook to fetch news data.
 *
 * @returns An object containing the news data, loading state, and error state.
 */
export const useNewsData = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNews = async () => {
      setIsLoading(true);
      try {
        // Simulate an API call with a delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Simulate a potential error for demonstration purposes
        if (Math.random() > 0.9) {
            throw new Error("Failed to fetch news data from the server.");
        }

        setNews(mockNewsData);
        setError(null);
      } catch (err) {
        if (err instanceof Error) {
            setError(err.message);
        } else {
            setError("An unknown error occurred while fetching news.");
        }
        setNews([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchNews();
  }, []); // Empty dependency array ensures this runs only once on mount

  return { news, isLoading, error };
};
