
import { useState, useEffect } from 'react';
import axios from 'axios';

// Define the shape of the historical data points
interface HistoricalData {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

// Define the state structure for the hook
interface UseHistoricalDataState {
    data: HistoricalData[];
    isLoading: boolean;
    error: string | null;
}

const API_BASE_URL = 'http://localhost:8000';

/**
 * Custom hook to fetch historical market data from the FastAPI backend.
 *
 * @param symbol The stock symbol to fetch data for.
 * @returns An object containing the data, loading state, and error state.
 */
export const useHistoricalData = (symbol: string): UseHistoricalDataState => {
    const [state, setState] = useState<UseHistoricalDataState>({
        data: [],
        isLoading: true,
        error: null,
    });

    useEffect(() => {
        const fetchHistoricalData = async () => {
            // Do not fetch if the symbol is empty
            if (!symbol) {
                setState({ data: [], isLoading: false, error: null });
                return;
            }

            setState({ data: [], isLoading: true, error: null });

            try {
                // Set date range for the last year
                const endDate = new Date().toISOString().split('T')[0];
                const startDate = new Date(new Date().setFullYear(new Date().getFullYear() - 1)).toISOString().split('T')[0];

                const response = await axios.get(
                    `${API_BASE_URL}/api/v1/market-data/historical/${symbol}`,
                    {
                        params: { start_date: startDate, end_date: endDate },
                    }
                );

                if (response.data) {
                    setState({ data: response.data, isLoading: false, error: null });
                }
            } catch (err) {
                let errorMessage = "An unknown error occurred.";
                if (axios.isAxiosError(err) && err.response) {
                    errorMessage = `Failed to fetch data: ${err.response.status} ${err.response.data.detail || err.message}`;
                } else if (err instanceof Error) {
                    errorMessage = err.message;
                }
                setState({ data: [], isLoading: false, error: errorMessage });
            }
        };

        fetchHistoricalData();
    }, [symbol]); // Re-run the effect when the symbol changes

    return state;
};
