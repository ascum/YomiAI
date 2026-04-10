const TAG = {
  "Cleora + BGE-M3": "bg-[#2e3257]/10 dark:bg-[#fffef7]/10 border-[#2e3257]/25 dark:border-[#fffef7]/25 text-[#2e3257] dark:text-[#fffef7]",
  "Cleora + CLIP":   "bg-[#627d9a]/10 dark:bg-[#627d9a]/20 border-[#627d9a]/30 text-[#627d9a] dark:text-[#babbbd]",
  "RL-DQN":          "bg-[#dfc5a4]/20 border-[#dfc5a4]/50 text-[#627d9a] dark:text-[#dfc5a4]",
  // legacy compat
  "Cleora + BLaIR":  "bg-[#2e3257]/10 dark:bg-[#fffef7]/10 border-[#2e3257]/25 dark:border-[#fffef7]/25 text-[#2e3257] dark:text-[#fffef7]",
};
const DEFAULT = "bg-[#babbbd]/15 border-[#babbbd]/30 text-[#627d9a] dark:text-[#babbbd]";

export function LayerTag({ label }) {
  return (
    <span className={`px-2 py-0.5 rounded-full border text-[9px] font-medium ${TAG[label] || DEFAULT}`}>
      {label}
    </span>
  );
}
