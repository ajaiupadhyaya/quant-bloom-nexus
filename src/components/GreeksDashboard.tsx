
import React, { useEffect, useRef, useState } from 'react';
import { usePortfolioGreeks } from '@/hooks/usePortfolioGreeks';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle, Loader2 } from 'lucide-react';

interface GreeksDashboardProps {
  // No props needed for now, but can be extended later
}

// Helper component for flashing effect
const FlashingValue: React.FC<{ value: number | string; prevValue: number | string | null }> = ({ value, prevValue }) => {
  const [flashClass, setFlashClass] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (prevValue !== null && value !== prevValue) {
      const numericValue = typeof value === 'number' ? value : parseFloat(value as string);
      const numericPrevValue = typeof prevValue === 'number' ? prevValue : parseFloat(prevValue as string);

      if (numericValue > numericPrevValue) {
        setFlashClass('flash-green');
      } else if (numericValue < numericPrevValue) {
        setFlashClass('flash-red');
      }

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        setFlashClass('');
      }, 500); // Flash for 500ms
    }
  }, [value, prevValue]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return <span className={flashClass}>{typeof value === 'number' ? value.toFixed(4) : value}</span>;
};

export const GreeksDashboard: React.FC<GreeksDashboardProps> = () => {
  const { data, isLoading, error } = usePortfolioGreeks(3000); // Fetch every 3 seconds
  const prevDataRef = useRef<typeof data>(null);

  useEffect(() => {
    prevDataRef.current = data;
  }, [data]);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-terminal-orange mb-4" />
          <p className="text-terminal-muted">Loading portfolio Greeks...</p>
          <Skeleton className="w-full h-48 mt-4" />
        </div>
      );
    }

    if (error) {
      return (
        <Alert variant="destructive" className="h-full flex flex-col justify-center items-center">
          <AlertTriangle className="h-6 w-6" />
          <AlertTitle>Error Loading Greeks</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      );
    }

    if (!data) {
      return (
        <div className="flex items-center justify-center h-full text-terminal-muted">
          <p>No portfolio Greeks data available.</p>
        </div>
      );
    }

    const greeks = [
      { name: 'Delta', value: data.total_delta, prevValue: prevDataRef.current?.total_delta },
      { name: 'Gamma', value: data.total_gamma, prevValue: prevDataRef.current?.total_gamma },
      { name: 'Vega', value: data.total_vega, prevValue: prevDataRef.current?.total_vega },
      { name: 'Theta', value: data.total_theta, prevValue: prevDataRef.current?.total_theta },
    ];

    return (
      <Table className="text-terminal-text">
        <TableHeader>
          <TableRow className="border-terminal-border">
            <TableHead className="text-terminal-orange">Greek</TableHead>
            <TableHead className="text-right text-terminal-orange">Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {greeks.map((greek) => (
            <TableRow key={greek.name} className="border-terminal-border/50">
              <TableCell className="font-medium text-terminal-cyan">{greek.name}</TableCell>
              <TableCell className="text-right font-mono">
                <FlashingValue value={greek.value} prevValue={greek.prevValue} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  };

  return (
    <div className="terminal-panel h-full flex flex-col p-4">
      <h2 className="text-lg font-semibold text-terminal-orange mb-4">Portfolio Greeks</h2>
      {renderContent()}
    </div>
  );
};
