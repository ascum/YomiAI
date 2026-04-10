import React from "react";

export function BentoGrid({ children, className = "" }) {
  return (
    <div className={`grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto px-6 py-16 ${className}`}>
      {children}
    </div>
  );
}

export function BentoItem({ children, className = "", span = "col-span-1" }) {
  return (
    <div className={`${span} ${className}`}>
      {children}
    </div>
  );
}
