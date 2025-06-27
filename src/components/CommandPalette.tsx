
import React, { useState, useEffect } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { Command as CommandPrimitive } from 'cmdk';
import { 
    File, 
    Search, 
    LayoutDashboard, 
    Briefcase, 
    ArrowRight, 
    BarChart2 
} from 'lucide-react';

// Mock stock data - in a real app, this would come from an API
const stockTickers = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corp.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'AMZN', name: 'Amazon.com, Inc.' },
    { symbol: 'TSLA', name: 'Tesla, Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.' },
    { symbol: 'META', name: 'Meta Platforms, Inc.' },
];

interface CommandPaletteProps {
    onClose: () => void;
    onSymbolSelect: (symbol: string) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ onClose, onSymbolSelect }) => {
    const [isOpen, setIsOpen] = useState(true);

    // Hotkey to open/close the palette
    useHotkeys('meta+k', (e) => {
        e.preventDefault();
        setIsOpen(prev => !prev);
    });

    useEffect(() => {
        if (!isOpen) {
            onClose();
        }
    }, [isOpen, onClose]);

    const handleSymbolSelect = (symbol: string) => {
        onSymbolSelect(symbol);
        setIsOpen(false);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-16">
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={() => setIsOpen(false)} />
            <CommandPrimitive className="relative z-10 w-full max-w-2xl rounded-lg border border-terminal-border bg-terminal-bg text-terminal-text shadow-2xl">
                <div className="flex items-center border-b border-terminal-border px-3">
                    <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
                    <CommandPrimitive.Input 
                        className="flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-terminal-muted disabled:cursor-not-allowed disabled:opacity-50"
                        placeholder="Type a command or search..."
                    />
                </div>
                <CommandPrimitive.List className="max-h-[400px] overflow-y-auto overflow-x-hidden">
                    <CommandPrimitive.Empty className="py-6 text-center text-sm">No results found.</CommandPrimitive.Empty>
                    
                    <CommandPrimitive.Group heading="Navigation" className="p-2 text-xs text-terminal-muted">
                        <CommandItem onSelect={() => console.log('Navigate to Dashboard')}>
                            <LayoutDashboard className="mr-2 h-4 w-4" />
                            <span>Go to Dashboard</span>
                        </CommandItem>
                        <CommandItem onSelect={() => console.log('Open Screener')}>
                            <BarChart2 className="mr-2 h-4 w-4" />
                            <span>Open Screener</span>
                        </CommandItem>
                        <CommandItem onSelect={() => console.log('New Trade')}>
                            <Briefcase className="mr-2 h-4 w-4" />
                            <span>New Trade</span>
                        </CommandItem>
                    </CommandPrimitive.Group>

                    <CommandPrimitive.Group heading="Stock Tickers" className="p-2 text-xs text-terminal-muted">
                        {stockTickers.map(stock => (
                            <CommandItem key={stock.symbol} onSelect={() => handleSymbolSelect(stock.symbol)}>
                                <File className="mr-2 h-4 w-4" />
                                <div className="flex w-full items-center justify-between">
                                    <span>{stock.name}</span>
                                    <span className="text-terminal-cyan">{stock.symbol}</span>
                                </div>
                            </CommandItem>
                        ))}
                    </CommandPrimitive.Group>
                </CommandPrimitive.List>
            </CommandPrimitive>
        </div>
    );
};

// A wrapper for the command item to provide consistent styling
const CommandItem = ({ children, ...props }: React.ComponentProps<typeof CommandPrimitive.Item>) => (
    <CommandPrimitive.Item 
        className="relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none aria-selected:bg-terminal-border aria-selected:text-terminal-orange data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
        {...props}
    >
        {children}
    </CommandPrimitive.Item>
);
