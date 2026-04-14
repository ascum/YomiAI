import { InfoTooltip } from "../../ui/InfoTooltip";

const BAR_META = [
  {
    label:   "CTR",
    tip:     "Ratio of positive interactions (clicks + carts) out of all session interactions.",
    formula: "(clicks + carts) / total",
    value: ({ clicks, total }) => total > 0 ? clicks / total : 0,
  },
  {
    label:   "Depth",
    tip:     "How deep into the session you've gone. Saturates at 20 interactions.",
    formula: "min(interactions / 20, 1)",
    value: ({ total }) => Math.min(total / 20, 1),
  },
  {
    label:   "Model Fit",
    tip:     "Proxy for how much positive training signal DIF-SASRec has received from you this session.",
    formula: "min(clicks / 10, 1) × 0.8 + 0.1",
    value: ({ clicks }) => Math.min(clicks / 10, 1) * 0.8 + 0.1,
  },
  {
    label:   "Diversity",
    tip:     "Fraction of interactions that were unique books — measures exploration vs revisiting.",
    formula: "unique items / total",
    value: ({ uniqueIds, total }) => uniqueIds / Math.max(total, 1),
  },
];

export function ProfileRadar({ interactions }) {
  const total     = interactions.length;
  const clicks    = interactions.filter(i => i.action === "click" || i.action === "cart").length;
  const uniqueIds = new Set(interactions.map(i => i.id)).size;

  const ctx = { total, clicks, uniqueIds };

  return (
    <div className="space-y-3.5">
      {BAR_META.map(b => {
        const pct = b.value(ctx) * 100;
        return (
          <div key={b.label}>
            {/* Label row */}
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-[#627d9a] dark:text-[#babbbd] flex items-center gap-0.5" style={{ fontSize: 10 }}>
                {b.label}
                <InfoTooltip tip={b.tip} formula={b.formula} />
              </span>
              <span className="font-mono font-semibold text-[#2e3257] dark:text-[#fffef7]/80 tabular-nums" style={{ fontSize: 10 }}>
                {Math.round(pct)}%
              </span>
            </div>

            {/* Bar track */}
            <div className="relative h-1.5 rounded-full overflow-hidden bg-[#babbbd]/40 dark:bg-[#627d9a]/25">
              {/* Subtle grid lines at 25/50/75% */}
              {[25, 50, 75].map(mark => (
                <div
                  key={mark}
                  className="absolute top-0 bottom-0 w-px bg-white/40 dark:bg-[#1a1b2e]/40"
                  style={{ left: `${mark}%` }}
                />
              ))}
              {/* Filled bar with gradient */}
              <div
                className="engagement-bar h-full rounded-full transition-all duration-700 ease-out"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
