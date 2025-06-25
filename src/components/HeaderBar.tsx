
import React from 'react';
import { Terminal, Bell, Settings, Layout, Shield, Monitor, Zap } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";

interface HeaderBarProps {
  currentTime: Date;
  notifications: number;
  layout: string;
  setLayout: (layout: string) => void;
}

export const HeaderBar = ({ currentTime, notifications, layout, setLayout }: HeaderBarProps) => {
  const layouts = [
    { id: 'professional', name: 'Professional', icon: Layout },
    { id: 'analytical', name: 'Analytical', icon: BarChart },
    { id: 'compact', name: 'Compact', icon: Monitor },
  ];

  return (
    <div className="h-10 flex items-center justify-between px-6 border-b border-terminal-border/50">
      <div className="flex items-center space-x-8">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Terminal className="w-6 h-6 text-terminal-orange animate-pulse-subtle" />
            <div className="absolute -top-1 -right-1 w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
          </div>
          <div>
            <span className="font-bold text-xl text-terminal-orange bg-gradient-to-r from-terminal-orange to-terminal-amber bg-clip-text text-transparent">
              BLOOMBERG TERMINAL
            </span>
            <div className="text-xs text-terminal-muted">Professional Edition v12.8.4</div>
          </div>
        </div>
        
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
            <span className="text-terminal-green font-medium">LIVE MARKET DATA</span>
          </div>
          <div className="text-terminal-muted">
            {currentTime.toLocaleTimeString()} EST
          </div>
          <div className="text-terminal-cyan font-mono">
            SPX: 4,521.23 <span className="text-terminal-green">+12.45</span>
          </div>
          <div className="text-terminal-cyan font-mono">
            NDX: 15,245.67 <span className="text-terminal-red">-8.23</span>
          </div>
        </div>
      </div>
      
      <div className="flex items-center space-x-3">
        {/* Function Key Shortcuts Display */}
        <div className="flex items-center space-x-2 text-xs text-terminal-muted">
          <kbd className="bg-terminal-border px-2 py-1 rounded">F8</kbd>
          <span>Equity</span>
          <kbd className="bg-terminal-border px-2 py-1 rounded">F9</kbd>
          <span>Bonds</span>
        </div>

        {/* Layout Selector */}
        <DropdownMenu>
          <DropdownMenuTrigger className="terminal-button flex items-center space-x-2 hover:glow-orange">
            <Layout className="w-4 h-4" />
            <span className="hidden md:inline">Layout</span>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
            <DropdownMenuLabel>Choose Layout</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {layouts.map((layoutOption) => (
              <DropdownMenuItem 
                key={layoutOption.id}
                onClick={() => setLayout(layoutOption.id)}
                className="hover:bg-terminal-border"
              >
                <layoutOption.icon className="w-4 h-4 mr-2" />
                {layoutOption.name}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Notifications */}
        <button className="terminal-button relative hover:glow-orange">
          <Bell className="w-4 h-4" />
          {notifications > 0 && (
            <span className="absolute -top-1 -right-1 bg-terminal-red text-xs rounded-full w-5 h-5 flex items-center justify-center text-white font-bold animate-pulse">
              {notifications}
            </span>
          )}
        </button>

        {/* Settings */}
        <DropdownMenu>
          <DropdownMenuTrigger className="terminal-button hover:glow-cyan">
            <Settings className="w-4 h-4" />
          </DropdownMenuTrigger>
          <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
            <DropdownMenuLabel>Terminal Settings</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="hover:bg-terminal-border">
              <Shield className="w-4 h-4 mr-2" />
              Security Settings
            </DropdownMenuItem>
            <DropdownMenuItem className="hover:bg-terminal-border">
              <Monitor className="w-4 h-4 mr-2" />
              Display Preferences
            </DropdownMenuItem>
            <DropdownMenuItem className="hover:bg-terminal-border">
              <Zap className="w-4 h-4 mr-2" />
              Performance Tuning
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
};
