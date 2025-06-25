
import React from 'react';
import { Command, Search } from 'lucide-react';

interface CommandBarProps {
  commandInput: string;
  setCommandInput: (input: string) => void;
  onCommandSubmit: (e: React.FormEvent) => void;
  setShowCommandPalette: (show: boolean) => void;
}

export const CommandBar = ({ 
  commandInput, 
  setCommandInput, 
  onCommandSubmit, 
  setShowCommandPalette 
}: CommandBarProps) => {
  return (
    <div className="h-6 flex items-center px-6 bg-gradient-to-r from-terminal-bg to-terminal-panel border-b border-terminal-border/30">
      <div className="flex items-center space-x-2 w-full">
        <Command className="w-3 h-3 text-terminal-orange" />
        <form onSubmit={onCommandSubmit} className="flex-1">
          <input
            type="text"
            value={commandInput}
            onChange={(e) => setCommandInput(e.target.value)}
            placeholder="Enter command (e.g., AAPL <Equity> GP for price graph)..."
            className="bg-transparent text-terminal-text placeholder-terminal-muted text-xs font-mono w-full focus:outline-none border-none"
          />
        </form>
        <button 
          onClick={() => setShowCommandPalette(true)}
          className="flex items-center space-x-1 text-xs text-terminal-muted hover:text-terminal-orange transition-colors"
        >
          <Search className="w-3 h-3" />
          <kbd className="bg-terminal-border px-1 rounded text-xs">⌘K</kbd>
        </button>
      </div>
    </div>
  );
};
