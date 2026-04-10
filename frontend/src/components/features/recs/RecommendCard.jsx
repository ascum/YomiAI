import { useState } from "react";
import { BookCover } from "../../ui/BookCover";
import { LayerTag } from "../../ui/LayerTag";

function parseAuthor(raw) {
  if (!raw) return "Unknown Author";
  if (typeof raw === "string" && raw.startsWith("{")) {
    try { const m = raw.match(/'name':\s*'([^']+)'/); if (m) return m[1]; } catch (_) {}
  }
  return raw;
}

export function RecommendCard({ book, onInteract, onAskAI, rank }) {
  const [showAI, setShowAI]     = useState(false);
  const [aiText, setAiText]     = useState("");
  const [aiLoading, setLoading] = useState(false);

  const handleAI = async (e) => {
    e.stopPropagation();
    if (showAI) { setShowAI(false); return; }
    setShowAI(true);
    if (!aiText) {
      setLoading(true);
      try   { const t = await onAskAI(book); setAiText(t); }
      catch (_) { setAiText("Failed to query AI helper."); }
      finally   { setLoading(false); }
    }
  };

  const cover  = book.cover_color || "#1e1b4b";
  const author = parseAuthor(book.author);
  const match  = Math.min(99, Math.max(60, Math.floor((book.score || 0.5) * 80 + 15)));

  return (
    <div className="flex gap-3 p-3 rounded-xl border border-[#babbbd] dark:border-[#627d9a]/60
                    bg-white/40 dark:bg-[#fffef7]/5
                    hover:border-[#dfc5a4] dark:hover:border-[#dfc5a4]/50
                    hover:bg-[#dfc5a4]/8 dark:hover:bg-[#dfc5a4]/5
                    transition-all duration-200 cursor-pointer shadow-sm">

      <div className="flex flex-col items-center gap-1.5">
        <span className="font-mono text-[#babbbd] dark:text-[#627d9a] font-bold" style={{ fontSize: 10 }}>
          #{rank + 1}
        </span>
        <BookCover color={cover} title={book.title} size="sm" imageUrl={book.image_url} />
      </div>

      <div className="flex-1 min-w-0">
        <p className="font-serif font-semibold leading-snug text-[#2e3257] dark:text-[#fffef7]" style={{ fontSize: 12 }}>
          {book.title}
        </p>
        <p className="text-[#627d9a] dark:text-[#babbbd] mt-0.5" style={{ fontSize: 10 }}>{author}</p>

        <div className="flex items-center gap-2 mt-1.5">
          {book.layer && <LayerTag label={book.layer} />}
          <span className="font-mono text-[#babbbd] dark:text-[#627d9a]" style={{ fontSize: 9 }}>
            match: {match}%
          </span>
        </div>

        {/* Row 1 */}
        <div className="flex gap-1.5 mt-2">
          <button
            onClick={() => onInteract(book, "click")}
            className="flex-1 py-1 rounded-lg text-[10px] transition-all duration-150
                       bg-[#2e3257]/10 dark:bg-[#fffef7]/10
                       border border-[#2e3257]/22 dark:border-[#fffef7]/18
                       text-[#2e3257] dark:text-[#fffef7]
                       hover:bg-[#dfc5a4]/30 hover:border-[#dfc5a4]"
          >
            ✓ Interested
          </button>
          <button
            onClick={() => onInteract(book, "skip")}
            className="flex-1 py-1 rounded-lg text-[10px] transition-all duration-150
                       bg-transparent border border-[#babbbd] dark:border-[#627d9a]/55
                       text-[#babbbd] dark:text-[#627d9a]
                       hover:border-[#dfc5a4] hover:text-[#627d9a] dark:hover:text-[#babbbd]"
          >
            ✗ Not for me
          </button>
        </div>

        {/* Row 2 */}
        <div className="flex gap-1.5 mt-1.5 relative">
          <button
            onClick={handleAI}
            className="flex-1 py-1 rounded-lg text-[10px] transition-all duration-150
                       bg-[#627d9a]/10 border border-[#627d9a]/25 dark:border-[#627d9a]/40
                       text-[#627d9a] dark:text-[#babbbd]
                       hover:bg-[#627d9a]/20"
          >
            {showAI ? "✕ Close AI" : "✦ Ask AI"}
          </button>
          <button
            onClick={e => { e.stopPropagation(); onInteract(book, "cart"); }}
            className="flex-1 py-1 rounded-lg text-[10px] font-medium transition-all duration-150
                       bg-emerald-50 dark:bg-emerald-900/20
                       border border-emerald-300 dark:border-emerald-700/60
                       text-emerald-700 dark:text-emerald-400
                       hover:bg-emerald-100 dark:hover:bg-emerald-900/40"
          >
            Add to Cart
          </button>

          {showAI && (
            <div className="absolute z-20 p-2.5 rounded-xl shadow-md fade-in text-left
                            bg-[#fffef7] dark:bg-[#2e3257]
                            border border-[#babbbd] dark:border-[#627d9a]"
              style={{ top: "calc(100% + 8px)", left: 0, right: -60, minWidth: 240 }}
            >
              <p className="font-serif font-bold text-[#2e3257] dark:text-[#fffef7] mb-0.5" style={{ fontSize: 12 }}>
                {book.title}
              </p>
              <p className="text-[#627d9a] dark:text-[#babbbd] mb-2" style={{ fontSize: 10 }}>by {author}</p>
              <div className="text-[#2e3257] dark:text-[#fffef7] overflow-y-auto" style={{ fontSize: 10, lineHeight: 1.55, whiteSpace: "pre-wrap", maxHeight: 140 }}>
                {aiLoading ? (
                  <span className="shimmer text-[#627d9a]">Thinking…</span>
                ) : (
                  aiText.split("\n").map((line, i) => (
                    <p key={i} className={line.startsWith("**") ? "mt-1.5" : ""}>
                      {line.split(/(\*\*.*?\*\*)/).map((part, j) =>
                        part.startsWith("**") && part.endsWith("**")
                          ? <strong key={j}>{part.slice(2, -2)}</strong>
                          : part
                      )}
                    </p>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
