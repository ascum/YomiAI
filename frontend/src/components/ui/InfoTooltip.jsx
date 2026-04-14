/**
 * InfoTooltip — a small ⓘ icon that shows a tooltip on hover.
 *
 * Usage:
 *   <InfoTooltip tip="Plain English explanation" formula="formula string" />
 *
 * Pure CSS (Tailwind group-hover) — no JS state, no library.
 */
export function InfoTooltip({ tip, formula }) {
  return (
    <span className="relative group inline-flex items-center ml-1">
      {/* Icon */}
      <span
        className="inline-flex items-center justify-center w-3 h-3 rounded-full
                   border border-[#babbbd] dark:border-[#627d9a]
                   text-[#babbbd] dark:text-[#627d9a]
                   cursor-default select-none"
        style={{ fontSize: 7, lineHeight: 1 }}
      >
        i
      </span>

      {/* Tooltip bubble */}
      <span
        className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50
                   w-48 px-3 py-2 rounded-xl shadow-lg
                   bg-[#2e3257] dark:bg-[#fffef7]
                   text-[#fffef7] dark:text-[#2e3257]
                   opacity-0 group-hover:opacity-100
                   transition-opacity duration-150"
        style={{ fontSize: 10 }}
      >
        <span className="block leading-relaxed">{tip}</span>
        {formula && (
          <span
            className="block mt-1 pt-1 border-t border-[#fffef7]/20 dark:border-[#2e3257]/20
                       font-mono text-[#dfc5a4] dark:text-[#627d9a]"
            style={{ fontSize: 9 }}
          >
            {formula}
          </span>
        )}
        {/* Arrow */}
        <span
          className="absolute top-full left-1/2 -translate-x-1/2
                     border-4 border-transparent border-t-[#2e3257] dark:border-t-[#fffef7]"
        />
      </span>
    </span>
  );
}
