
import { useState, useEffect } from 'react';
import axios from 'axios';

interface SentimentResult {
  headline: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}

interface UseNewsSentimentState {
  data: SentimentResult[];
  isLoading: boolean;
  error: string | null;
}

const API_BASE_URL = 'http://localhost:8000';

/**
 * Custom hook to fetch news sentiment data from the FastAPI backend.
 * It polls the API at a specified interval.
 *
 * @param interval The polling interval in milliseconds.
 * @returns An object containing the data, loading state, and error state.
 */
export const useNewsSentiment = (interval: number = 10000): UseNewsSentimentState => {
  const [state, setState] = useState<UseNewsSentimentState>({
    data: [],
    isLoading: true,
    error: null,
  });

  useEffect(() => {
    const fetchNewsSentiment = async () => {
      try {
        const response = await axios.get<SentimentResult[]>(`${API_BASE_URL}/api/v1/news/sentiment`);
        setState(prevState => ({
          ...prevState,
          data: response.data,
          isLoading: false,
          error: null,
        }));
      } catch (err) {
        let errorMessage = "An unknown error occurred while fetching news sentiment.";
        if (axios.isAxiosError(err) && err.response) {
          errorMessage = `Failed to fetch news sentiment: ${err.response.status} ${err.response.data.detail || err.message}`;
        } else if (err instanceof Error) {
          errorMessage = err.message;
        }
        setState(prevState => ({
          ...prevState,
          data: [],
          isLoading: false,
          error: errorMessage,
        }));
      }
    };

    // Fetch immediately on mount
    fetchNewsSentiment();

    // Set up polling
    const intervalId = setInterval(fetchNewsSentiment, interval);

    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, [interval]);

  return state;
};
