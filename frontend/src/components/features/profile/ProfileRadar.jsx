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
    <div className="space-y-2.5">
      {BAR_META.map(b => (
        <div key={b.label} className="flex items-center gap-3">
          <span
            className="font-mono text-[#babbbd] dark:text-[#627d9a] flex items-center"
            style={{ fontSize: 10, width: 64, justifyContent: "flex-end" }}
          >
            {b.label}
            <InfoTooltip tip={b.tip} formula={b.formula} />
          </span>
          <div className="flex-1 h-1.5 rounded-full bg-[#babbbd]/30 dark:bg-[#627d9a]/25">
            <div
              className="h-1.5 rounded-full bg-[#2e3257] dark:bg-[#dfc5a4] transition-all duration-700"
              style={{ width: `${b.value(ctx) * 100}%` }}
            />
          </div>
          <span className="font-mono text-[#babbbd] dark:text-[#627d9a] w-8" style={{ fontSize: 10 }}>
            {Math.round(b.value(ctx) * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
