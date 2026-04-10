import React from "react";
import { Card } from "../../ui/Card";
import { Button } from "../../ui/Button";

export function BookCard({ book, onAction }) {
  if (!book) return null;

  return (
    <Card className="flex flex-col h-full hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors">
      <div className="relative aspect-[2/3] mb-6 bg-zinc-100 dark:bg-zinc-800 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800">
        {book.image_url ? (
          <img 
            src={book.image_url} 
            alt={book.title} 
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-zinc-400">
            No Cover
          </div>
        )}
      </div>
      
      <div className="flex-1">
        <h3 className="font-bold text-zinc-900 dark:text-zinc-50 text-xl line-clamp-2 mb-2 leading-tight">
          {book.title || "Untitled"}
        </h3>
        <p className="text-base text-zinc-500 dark:text-zinc-400 line-clamp-1 mb-6">
          {book.author_name || "Unknown Author"}
        </p>
      </div>

      <div className="flex gap-3 mt-auto pt-6 border-t border-zinc-100 dark:border-zinc-800">
        <Button 
          variant="primary" 
          className="flex-1 text-base"
          onClick={() => onAction(book.parent_asin, "cart")}
        >
          Add to Cart
        </Button>
        <Button 
          variant="secondary" 
          className="text-base"
          onClick={() => onAction(book.parent_asin, "click")}
        >
          View
        </Button>
      </div>
    </Card>
  );
}
