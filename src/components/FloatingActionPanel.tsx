
import React from 'react';
import { Brain, Target, Activity } from 'lucide-react';

export const FloatingActionPanel = () => {
  return (
    <div className="fixed bottom-8 right-4 flex flex-col space-y-2">
      <button className="bg-terminal-orange hover:bg-terminal-amber text-terminal-bg p-3 rounded-full shadow-2xl glow-orange transition-all duration-300 hover:scale-110 group">
        <Brain className="w-5 h-5" />
        <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
          AI Analysis
        </div>
      </button>
      <button className="bg-terminal-cyan hover:bg-terminal-cyan/80 text-terminal-bg p-3 rounded-full shadow-2xl glow-cyan transition-all duration-300 hover:scale-110 group">
        <Target className="w-5 h-5" />
        <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
          Quick Trade
        </div>
      </button>
      <button className="bg-terminal-green hover:bg-terminal-green/80 text-terminal-bg p-3 rounded-full shadow-2xl transition-all duration-300 hover:scale-110 group">
        <Activity className="w-5 h-5" />
        <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
          Live Activity
        </div>
      </button>
    </div>
  );
};
