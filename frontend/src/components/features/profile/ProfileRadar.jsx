export function ProfileRadar({ interactions }) {
  const total  = interactions.length;
  const clicks = interactions.filter(i => i.action === "click" || i.action === "cart").length;
  const ctr    = total > 0 ? clicks / total : 0;

  const bars = [
    { label: "CTR",       value: ctr },
    { label: "Depth",     value: Math.min(total / 20, 1) },
    { label: "RL Fit",    value: Math.min(clicks / 10, 1) * 0.8 + 0.1 },
    { label: "Diversity", value: new Set(interactions.map(i => i.id)).size / Math.max(total, 1) },
  ];

  return (
    <div className="space-y-2.5">
      {bars.map(b => (
        <div key={b.label} className="flex items-center gap-3">
          <span className="font-mono text-[#babbbd] dark:text-[#627d9a] w-14 text-right" style={{ fontSize: 10 }}>
            {b.label}
          </span>
          <div className="flex-1 h-1.5 rounded-full bg-[#babbbd]/30 dark:bg-[#627d9a]/25">
            <div
              className="h-1.5 rounded-full bg-[#2e3257] dark:bg-[#dfc5a4] transition-all duration-700"
              style={{ width: `${b.value * 100}%` }}
            />
          </div>
          <span className="font-mono text-[#babbbd] dark:text-[#627d9a] w-8" style={{ fontSize: 10 }}>
            {Math.round(b.value * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
