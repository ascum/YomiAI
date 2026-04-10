import React from "react";

export const Card = ({ children, className = "" }) => {
  return (
    <div className={`bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm rounded-xl p-6 ${className}`}>
      {children}
    </div>
  );
};
