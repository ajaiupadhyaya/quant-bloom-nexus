
import React from 'react';

interface StatusBarProps {
  currentTime: Date;
}

export const StatusBar = ({ currentTime }: StatusBarProps) => {
  return (
    <div className="fixed bottom-0 left-0 right-0 h-6 bg-terminal-panel border-t border-terminal-border flex items-center justify-between px-4 text-xs text-terminal-muted">
      <div className="flex items-center space-x-4">
        <span>Connected to Market Data Feed</span>
        <span className="text-terminal-green">●</span>
        <span>Last Update: {currentTime.toLocaleTimeString()}</span>
      </div>
      <div className="flex items-center space-x-4">
        <span>CPU: 12%</span>
        <span>Memory: 2.1GB</span>
        <span>Latency: 0.8ms</span>
      </div>
    </div>
  );
};
