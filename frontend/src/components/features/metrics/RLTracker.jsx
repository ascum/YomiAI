import React from "react";
import { Card } from "../../ui/Card";

export function RLTracker({ metrics }) {
  const { loss = 0, steps = 0, reward = 0 } = metrics || {};

  return (
    <Card>
      <div className="flex items-center justify-between mb-8">
        <h3 className="font-bold text-zinc-900 dark:text-zinc-50 text-xl tracking-tight">RL Performance</h3>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Active</span>
        </div>
      </div>

      <div className="space-y-8">
        <div>
          <div className="flex justify-between items-end mb-2">
            <span className="text-sm font-medium text-zinc-500 dark:text-zinc-400">Loss Convergence</span>
            <span className="text-xl font-mono font-bold text-zinc-900 dark:text-zinc-50">
              {loss.toFixed(4)}
            </span>
          </div>
          <div className="w-full h-2 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-black dark:bg-white transition-all duration-500" 
              style={{ width: `${Math.min(100, (1 - loss) * 100)}%` }}
            ></div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="p-4 bg-zinc-50 dark:bg-zinc-800/50 rounded-xl border border-zinc-100 dark:border-zinc-800">
            <span className="block text-xs font-bold text-zinc-400 uppercase tracking-widest mb-2">
              Steps
            </span>
            <span className="text-xl font-mono font-bold text-zinc-900 dark:text-zinc-50">
              {steps}
            </span>
          </div>
          <div className="p-4 bg-zinc-50 dark:bg-zinc-800/50 rounded-xl border border-zinc-100 dark:border-zinc-800">
            <span className="block text-xs font-bold text-zinc-400 uppercase tracking-widest mb-2">
              Reward
            </span>
            <span className={`text-xl font-mono font-bold ${reward >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
              {reward >= 0 ? '+' : ''}{reward.toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}
