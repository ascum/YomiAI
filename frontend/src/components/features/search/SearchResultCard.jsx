import { useState } from "react";
import { BookCover } from "../../ui/BookCover";
import { ScoreBadge } from "../../ui/ScoreBadge";

function parseAuthor(raw) {
  if (!raw) return "Unknown Author";
  if (typeof raw === "string" && raw.startsWith("{")) {
    try { const m = raw.match(/'name':\s*'([^']+)'/); if (m) return m[1]; } catch (_) {}
  }
  return raw;
}

export function SearchResultCard({ book, onInteract, onAskAIStream }) {
  const [showAI, setShowAI]     = useState(false);
  const [aiText, setAiText]     = useState("");
  const [aiLoading, setLoading] = useState(false);

  const handleAI = async (e) => {
    e.stopPropagation();
    if (showAI) { setShowAI(false); return; }
    setShowAI(true);
    if (!aiText) {
      setLoading(true);
      setAiText("");
      try {
        const stream = onAskAIStream(book);
        for await (const chunk of stream) {
          setLoading(false); // Hide loading as soon as first token arrives
          setAiText(prev => prev + chunk);
        }
      } catch (_) {
        setAiText("Failed to query AI helper.");
        setLoading(false);
      }
    }
  };

  const genre  = book.genre || "Books";
  const cover  = book.cover_color || "#1e1b4b";
  const author = parseAuthor(book.author);

  return (
    <div className="group flex gap-3 p-3 rounded-xl border border-[#babbbd] dark:border-[#627d9a]/60
                    bg-white/50 dark:bg-[#fffef7]/5
                    hover:border-[#dfc5a4] dark:hover:border-[#dfc5a4]/60
                    hover:bg-[#dfc5a4]/8 dark:hover:bg-[#dfc5a4]/5
                    transition-all duration-200 cursor-pointer shadow-sm">

      <BookCover color={cover} title={book.title} size="md" imageUrl={book.image_url} />

      <div className="flex-1 min-w-0">
        {/* Title row */}
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-serif font-semibold leading-tight text-[#2e3257] dark:text-[#fffef7]" style={{ fontSize: 13 }}>
              {book.title}
            </p>
            <p className="text-[#627d9a] dark:text-[#babbbd] mt-0.5" style={{ fontSize: 11 }}>{author}</p>
            <span className="inline-block mt-1 px-2 py-0.5 rounded-full text-[9px]
                             bg-[#dfc5a4]/25 text-[#627d9a] dark:text-[#babbbd]">
              {genre}
            </span>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <ScoreBadge score={book.score} label="Match" />
            {book.text_sim > 0 && <ScoreBadge score={book.text_sim} label="Text" />}
            {book.img_sim  > 0 && <ScoreBadge score={book.img_sim}  label="Img"  />}
          </div>
        </div>

        {/* Action row 1 */}
        <div className="flex gap-2 mt-2.5">
          <button
            onClick={() => onInteract(book, "click")}
            className="flex-1 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-150
                       bg-[#2e3257]/10 dark:bg-[#fffef7]/10
                       border border-[#2e3257]/25 dark:border-[#fffef7]/20
                       text-[#2e3257] dark:text-[#fffef7]
                       hover:bg-[#dfc5a4]/30 hover:border-[#dfc5a4]"
          >
            ✓ Interested
          </button>
          <button
            onClick={() => onInteract(book, "skip")}
            className="flex-1 py-1.5 rounded-lg text-[11px] transition-all duration-150
                       bg-transparent border border-[#babbbd] dark:border-[#627d9a]/60
                       text-[#babbbd] dark:text-[#627d9a]
                       hover:border-[#dfc5a4] hover:text-[#627d9a] dark:hover:text-[#babbbd]"
          >
            ✕ Skip
          </button>
        </div>

        {/* Action row 2 */}
        <div className="flex gap-2 mt-1.5 relative">
          <button
            onClick={handleAI}
            className="flex-1 py-1.5 rounded-lg text-[11px] transition-all duration-150
                       bg-[#627d9a]/10 dark:bg-[#627d9a]/15
                       border border-[#627d9a]/25 dark:border-[#627d9a]/40
                       text-[#627d9a] dark:text-[#babbbd]
                       hover:bg-[#627d9a]/20"
          >
            {showAI ? "✕ Close AI" : "✦ Ask AI"}
          </button>
          <button
            onClick={e => { e.stopPropagation(); onInteract(book, "cart"); }}
            className="flex-1 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-150
                       bg-emerald-50 dark:bg-emerald-900/20
                       border border-emerald-300 dark:border-emerald-700/60
                       text-emerald-700 dark:text-emerald-400
                       hover:bg-emerald-100 dark:hover:bg-emerald-900/40"
          >
            Add to Cart
          </button>

          {/* AI popover */}
          {showAI && (
            <div className="absolute z-20 p-3 rounded-xl shadow-md fade-in text-left
                            bg-[#fffef7] dark:bg-[#2e3257]
                            border border-[#babbbd] dark:border-[#627d9a]"
              style={{ top: "calc(100% + 8px)", left: 0, right: 0, minWidth: 260 }}
            >
              <p className="font-serif font-bold text-[#2e3257] dark:text-[#fffef7] mb-0.5" style={{ fontSize: 14 }}>
                {book.title}
              </p>
              <p className="text-[#627d9a] dark:text-[#babbbd] mb-2" style={{ fontSize: 11 }}>by {author}</p>
              <div className="text-[#2e3257] dark:text-[#fffef7] overflow-y-auto" style={{ fontSize: 11, lineHeight: 1.55, whiteSpace: "pre-wrap", maxHeight: 160 }}>
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
