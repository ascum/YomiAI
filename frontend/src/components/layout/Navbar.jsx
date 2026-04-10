import React from "react";
import { Button } from "../ui/Button";

export default function Navbar({ isDark, setIsDark }) {
  return (
    <nav className="sticky top-0 z-50 backdrop-blur-md bg-white/70 dark:bg-zinc-950/70 border-b border-zinc-200 dark:border-zinc-800 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 bg-black dark:bg-white rounded-xl flex items-center justify-center">
            <span className="text-white dark:text-black font-black text-2xl leading-none">A</span>
          </div>
          <h1 className="text-zinc-900 dark:text-zinc-100 font-extrabold tracking-tight text-2xl ml-2">
            Multimodal AI
          </h1>
        </div>
        
        <div className="flex items-center gap-8">
          <div className="hidden md:flex gap-8">
            <button
              onClick={() => setIsDark(!isDark)}
              className="text-base font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors"
            >
              {isDark ? "Light Mode" : "Dark Mode"}
            </button>
            <a href="#" className="text-base font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 transition-colors">
              Docs
            </a>
          </div>
          <div className="h-6 w-[1px] bg-zinc-200 dark:bg-zinc-800 hidden md:block"></div>
          <Button variant="primary" className="text-base px-6">Get Started</Button>
        </div>
      </div>
    </nav>
  );
}
