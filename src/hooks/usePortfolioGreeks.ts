
import { useState, useEffect } from 'react';
import axios from 'axios';

interface PortfolioGreeks {
  total_delta: number;
  total_gamma: number;
  total_vega: number;
  total_theta: number;
}

interface UsePortfolioGreeksState {
  data: PortfolioGreeks | null;
  isLoading: boolean;
  error: string | null;
}

const API_BASE_URL = 'http://localhost:8000';

/**
 * Custom hook to fetch portfolio Greeks data from the FastAPI backend.
 * It polls the API at a specified interval.
 *
 * @param interval The polling interval in milliseconds.
 * @returns An object containing the data, loading state, and error state.
 */
export const usePortfolioGreeks = (interval: number = 3000): UsePortfolioGreeksState => {
  const [state, setState] = useState<UsePortfolioGreeksState>({
    data: null,
    isLoading: true,
    error: null,
  });

  useEffect(() => {
    const fetchGreeks = async () => {
      try {
        const response = await axios.get<PortfolioGreeks>(`${API_BASE_URL}/api/v1/portfolio/greeks`);
        setState(prevState => ({
          ...prevState,
          data: response.data,
          isLoading: false,
          error: null,
        }));
      } catch (err) {
        let errorMessage = "An unknown error occurred while fetching Greeks.";
        if (axios.isAxiosError(err) && err.response) {
          errorMessage = `Failed to fetch Greeks: ${err.response.status} ${err.response.data.detail || err.message}`;
        } else if (err instanceof Error) {
          errorMessage = err.message;
        }
        setState(prevState => ({
          ...prevState,
          data: null,
          isLoading: false,
          error: errorMessage,
        }));
      }
    };

    // Fetch immediately on mount
    fetchGreeks();

    // Set up polling
    const intervalId = setInterval(fetchGreeks, interval);

    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, [interval]);

  return state;
};
