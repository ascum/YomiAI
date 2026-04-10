const P = "animate-pulse rounded bg-[#babbbd]/30 dark:bg-[#627d9a]/20";

export function SkeletonCard({ size = "md" }) {
  if (size === "sm") {
    return (
      <div className="flex gap-3 p-3 rounded-xl border border-[#babbbd]/40 dark:border-[#627d9a]/25">
        <div className={`${P} flex-shrink-0`} style={{ width: 40, height: 56 }} />
        <div className="flex-1 space-y-2 pt-1">
          <div className={`${P} h-3 w-3/4`} />
          <div className={`${P} h-2.5 w-1/2`} />
          <div className={`${P} h-2 w-1/3`} />
          <div className="flex gap-1.5 pt-1">
            <div className={`${P} flex-1 h-6`} />
            <div className={`${P} flex-1 h-6`} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 p-3 rounded-xl border border-[#babbbd]/40 dark:border-[#627d9a]/25">
      <div className={`${P} flex-shrink-0`} style={{ width: 64, height: 88 }} />
      <div className="flex-1 space-y-2 pt-1">
        <div className={`${P} h-3.5 w-3/4`} />
        <div className={`${P} h-3 w-1/2`} />
        <div className={`${P} h-2.5 w-1/4`} />
        <div className="flex gap-2 pt-2">
          <div className={`${P} flex-1 h-7`} />
          <div className={`${P} flex-1 h-7`} />
        </div>
        <div className="flex gap-2">
          <div className={`${P} flex-1 h-7`} />
          <div className={`${P} flex-1 h-7`} />
        </div>
      </div>
    </div>
  );
}
