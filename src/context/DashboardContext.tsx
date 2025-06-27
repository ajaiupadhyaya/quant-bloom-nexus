
import React, { createContext, useState, useContext, ReactNode } from 'react';

// Define the shape of the context state
interface DashboardContextType {
  selectedSymbol: string;
  setSelectedSymbol: (symbol: string) => void;
}

// Create the context with a default value
const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

// Create a provider component
export const DashboardProvider = ({ children }: { children: ReactNode }) => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');

  return (
    <DashboardContext.Provider value={{ selectedSymbol, setSelectedSymbol }}>
      {children}
    </DashboardContext.Provider>
  );
};

// Create a custom hook to use the dashboard context
export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
};
