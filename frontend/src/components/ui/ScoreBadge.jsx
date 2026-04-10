// Stroke colors reference CSS variables defined in index.css so they adapt to dark mode
// without needing JS-level theme awareness.
const stroke = (pct) => {
  if (pct > 85) return "var(--score-high)";
  if (pct > 70) return "var(--score-mid)";
  return "var(--score-low)";
};

export function ScoreBadge({ score, label }) {
  const pct = Math.round(score * 100);
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-10 h-10">
        <svg className="w-10 h-10 -rotate-90" viewBox="0 0 36 36">
          <circle
            cx="18" cy="18" r="14"
            fill="none"
            stroke="var(--score-track)"
            strokeWidth="2.5"
          />
          <circle
            cx="18" cy="18" r="14"
            fill="none"
            stroke={stroke(pct)}
            strokeWidth="2.5"
            strokeDasharray={`${pct * 0.88} 88`}
            strokeLinecap="round"
          />
        </svg>
        <span
          className="absolute inset-0 flex items-center justify-center font-mono text-[#2e3257] dark:text-[#fffef7]"
          style={{ fontSize: 9 }}
        >
          {pct}
        </span>
      </div>
      <span className="text-[#babbbd] dark:text-[#627d9a] mt-0.5" style={{ fontSize: 9 }}>
        {label}
      </span>
    </div>
  );
}
