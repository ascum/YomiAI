import React, { useRef } from "react";
import { Card } from "../../ui/Card";
import { Button } from "../../ui/Button";

export function HybridSearch({ 
  onSearch, 
  onImageUpload, 
  loading, 
  currentImage 
}) {
  const fileInputRef = useRef(null);

  return (
    <Card className="p-0 overflow-hidden">
      <div className="p-8 border-b border-zinc-100 dark:border-zinc-800">
        <h3 className="font-bold text-zinc-900 dark:text-zinc-50 text-xl mb-6">Multimodal Search</h3>
        <div className="flex gap-3">
          <input 
            type="text" 
            placeholder="Search by plot, theme, or author..."
            className="flex-1 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg px-4 py-3 text-base text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-black/5 dark:focus:ring-white/5 transition-all"
            onKeyDown={(e) => {
              if (e.key === "Enter") onSearch(e.target.value);
            }}
          />
          <Button variant="primary" disabled={loading} className="px-6 text-base">
            {loading ? "..." : "Search"}
          </Button>
        </div>
      </div>

      <div className="p-8 bg-zinc-50/50 dark:bg-zinc-900/50">
        <span className="block text-xs font-bold text-zinc-400 uppercase tracking-widest mb-4">
          Visual Query (CLIP)
        </span>
        
        <div 
          onClick={() => fileInputRef.current?.click()}
          className={`
            border-2 border-dashed rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all
            ${currentImage ? 'border-black dark:border-white bg-white dark:bg-zinc-800 shadow-sm' : 'border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-500 hover:bg-white dark:hover:bg-zinc-800'}
          `}
        >
          <input 
            type="file" 
            className="hidden" 
            ref={fileInputRef}
            onChange={(e) => {
              const file = e.target.files[0];
              if (file) {
                const reader = new FileReader();
                reader.onloadend = () => onImageUpload(reader.result.split(",")[1]);
                reader.readAsDataURL(file);
              }
            }}
          />
          
          {currentImage ? (
            <div className="w-32 h-32 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-700 mb-4 shadow-md">
              <img 
                src={`data:image/jpeg;base64,${currentImage}`} 
                alt="Upload preview" 
                className="w-full h-full object-cover"
              />
            </div>
          ) : (
            <div className="w-14 h-14 rounded-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 flex items-center justify-center mb-4 shadow-sm">
              <svg className="w-7 h-7 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
          )}
          
          <p className="text-base font-bold text-zinc-900 dark:text-zinc-50">
            {currentImage ? "Change Image" : "Drop book cover"}
          </p>
          <p className="text-sm text-zinc-400 mt-1">
            PNG, JPG up to 5MB
          </p>
        </div>
      </div>
    </Card>
  );
}
