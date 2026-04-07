import { useState, useEffect, useRef, useCallback } from "react";

// ─── API CONFIG ───────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

async function apiSearch(query, imageBase64 = null, sessionId = null) {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, image_base64: imageBase64, top_k: 20, session_id: sessionId }),
  });
  return res.json();
}

async function apiRecommend(userId, sessionId = null) {
  const res = await fetch(`${API_BASE}/recommend?user_id=${userId}&session_id=${sessionId}`);
  return res.json();
}

async function apiInteract(userId, itemId, action, sessionId = null) {
  const res = await fetch(`${API_BASE}/interact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, item_id: itemId, action, session_id: sessionId }),
  });
  return res.json();
}

async function apiProfile(userId) {
  const res = await fetch(`${API_BASE}/profile?user_id=${userId}`);
  return res.json();
}

async function apiRlMetrics(userId) {
  const res = await fetch(`${API_BASE}/rl_metrics?user_id=${userId}`);
  return res.json();
}

async function apiAskLLM(title, author, userPrompt) {
  const res = await fetch(`${API_BASE}/ask_llm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item_id: "preview", title, author, user_prompt: userPrompt }),
  });
  return res.json();
}

// ─── MOCK DATA (used when backend not connected) ──────────────────────────────
const MOCK_BOOKS = [
  { id: "b001", title: "The Name of the Wind", author: "Patrick Rothfuss", genre: "Dark Fantasy", score: 0.94, cover_color: "#1a0a2e", text_sim: 0.91, img_sim: 0.88 },
  { id: "b002", title: "Mistborn: The Final Empire", author: "Brandon Sanderson", genre: "Epic Fantasy", score: 0.89, cover_color: "#0d1b2a", text_sim: 0.85, img_sim: 0.82 },
  { id: "b003", title: "The Way of Kings", author: "Brandon Sanderson", genre: "High Fantasy", score: 0.87, cover_color: "#162040", text_sim: 0.83, img_sim: 0.79 },
  { id: "b004", title: "A Shadow in the Ember", author: "Jennifer L. Armentrout", genre: "Dark Romance Fantasy", score: 0.82, cover_color: "#2d0a0a", text_sim: 0.78, img_sim: 0.81 },
  { id: "b005", title: "Blood and Ash", author: "Jennifer L. Armentrout", genre: "Dark Fantasy", score: 0.80, cover_color: "#1a0000", text_sim: 0.76, img_sim: 0.83 },
];

const MOCK_RECS = {
  people_also_buy: [
    { id: "r001", title: "The Lies of Locke Lamora", author: "Scott Lynch", genre: "Fantasy Heist", score: 0.91, cover_color: "#0a1a2d", layer: "Cleora + BLaIR" },
    { id: "r002", title: "Assassin's Apprentice", author: "Robin Hobb", genre: "Epic Fantasy", score: 0.88, cover_color: "#0d2010", layer: "Cleora + CLIP" },
  ],
  you_might_like: [
    { id: "r003", title: "The Black Prism", author: "Brent Weeks", genre: "Dark Fantasy", score: 0.86, cover_color: "#1a1000", layer: "RL-DQN" },
    { id: "r004", title: "Kushiel's Dart", author: "Jacqueline Carey", genre: "Fantasy", score: 0.84, cover_color: "#100808", layer: "RL-DQN" },
  ]
};

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function BookCover({ color, title, size = "md", imageUrl }) {
  const dims = {
    lg: { width: 80, height: 112 },
    md: { width: 64, height: 88 },
    sm: { width: 40, height: 56 },
  };
  const { width, height } = dims[size] || dims.md;
  return (
    <div
      className={`rounded-sm flex-shrink-0 relative overflow-hidden ${imageUrl ? "book-cover-uninvert" : ""}`}
      style={{
        background: imageUrl ? `url(${imageUrl}) center/cover no-repeat` : `linear-gradient(135deg, ${color}ee, ${color}88)`,
        boxShadow: `2px 3px 12px ${color}66, inset -2px 0 6px rgba(0,0,0,0.4)`,
        width,
        height,
      }}
    >
      {!imageUrl && (
        <div className="absolute inset-0" style={{ background: "linear-gradient(to right, rgba(255,255,255,0.08) 0%, transparent 30%)" }} />
      )}
      {!imageUrl && (
        <div className="absolute bottom-0 left-0 right-0 p-1">
          <div className="text-white/60 font-mono" style={{ fontSize: 5, lineHeight: 1.3 }}>
            {title?.slice(0, 20)}
          </div>
        </div>
      )}
    </div>
  );
}

function ScoreBadge({ score, label }) {
  const pct = Math.round(score * 100);
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-10 h-10">
        <svg className="w-10 h-10 -rotate-90" viewBox="0 0 36 36">
          <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="3" />
          <circle
            cx="18" cy="18" r="14" fill="none"
            stroke={pct > 85 ? "#a78bfa" : pct > 70 ? "#60a5fa" : "#94a3b8"}
            strokeWidth="3"
            strokeDasharray={`${pct * 0.88} 88`}
            strokeLinecap="round"
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-white font-mono" style={{ fontSize: 9 }}>{pct}</span>
      </div>
      <span className="text-white/40 mt-0.5" style={{ fontSize: 9 }}>{label}</span>
    </div>
  );
}

function LayerTag({ label }) {
  const colors = {
    "Cleora + BLaIR": "rgba(167,139,250,0.15)",
    "Cleora + CLIP": "rgba(96,165,250,0.15)",
    "RL-DQN": "rgba(52,211,153,0.15)",
  };
  const border = {
    "Cleora + BLaIR": "rgba(167,139,250,0.4)",
    "Cleora + CLIP": "rgba(96,165,250,0.4)",
    "RL-DQN": "rgba(52,211,153,0.4)",
  };
  return (
    <span
      className="text-white/70 px-2 py-0.5 rounded-full"
      style={{
        fontSize: 9,
        background: colors[label] || "rgba(255,255,255,0.05)",
        border: `1px solid ${border[label] || "rgba(255,255,255,0.1)"}`,
      }}
    >
      {label}
    </span>
  );
}

function ProfileRadar({ interactions }) {
  const total = interactions.length;
  const clicks = interactions.filter(i => i.action === "click" || i.action === "cart").length;
  const ctr = total > 0 ? clicks / total : 0;
  const bars = [
    { label: "CTR", value: ctr },
    { label: "Depth", value: Math.min(total / 20, 1) },
    { label: "RL Fit", value: Math.min(clicks / 10, 1) * 0.8 + 0.1 },
    { label: "Diversity", value: new Set(interactions.map(i => i.id)).size / Math.max(total, 1) },
  ];
  return (
    <div className="space-y-2">
      {bars.map(b => (
        <div key={b.label} className="flex items-center gap-3">
          <span className="text-white/40 w-14 text-right font-mono" style={{ fontSize: 10 }}>{b.label}</span>
          <div className="flex-1 h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
            <div
              className="h-1.5 rounded-full transition-all duration-700"
              style={{ width: `${b.value * 100}%`, background: "linear-gradient(to right, #7c3aed, #a78bfa)" }}
            />
          </div>
          <span className="text-white/30 font-mono w-8" style={{ fontSize: 10 }}>{Math.round(b.value * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

function SearchResultCard({ book, onInteract, onAskAI, isNew }) {
  const [showAI, setShowAI] = useState(false);
  const [aiText, setAiText] = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const handleAI = async (e) => {
    e.stopPropagation();
    if (showAI) {
      setShowAI(false);
      return;
    }
    setShowAI(true);
    if (!aiText) {
      setAiLoading(true);
      try {
        const text = await onAskAI(book);
        setAiText(text);
      } catch (err) {
        setAiText("Failed to query AI helper.");
      } finally {
        setAiLoading(false);
      }
    }
  };

  // Safe defaults & Parsers for minimal payload
  const displayGenre = book.genre || "Books";
  const displayCoverColor = book.cover_color || "#1e1b4b";
  
  // Failsafe for unparsed author dicts
  let displayAuthor = book.author || "Unknown Author";
  if (typeof displayAuthor === 'string' && displayAuthor.startsWith('{')) {
    try {
      // Try to extract name: 'Name' from stringified dict
      const match = displayAuthor.match(/'name':\s*'([^']+)'/);
      if (match) displayAuthor = match[1];
    } catch(e) {}
  }

  return (
    <div
      className={`flex gap-3 p-3 rounded-lg cursor-pointer group transition-all duration-300`}
      style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
      onMouseEnter={e => e.currentTarget.style.background = "rgba(167,139,250,0.08)"}
      onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.04)"}
    >
      <BookCover color={displayCoverColor} title={book.title} size="md" imageUrl={book.image_url} />
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="text-white/90 font-medium leading-tight" style={{ fontFamily: "'Playfair Display', Georgia, serif", fontSize: 13 }}>{book.title}</p>
            <p className="text-white/40 mt-0.5" style={{ fontSize: 11 }}>{displayAuthor}</p>
            <span className="inline-block mt-1 px-2 py-0.5 rounded-full text-white/50" style={{ fontSize: 9, background: "rgba(255,255,255,0.06)" }}>{displayGenre}</span>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <ScoreBadge score={book.score} label="Match" />
            {book.text_sim !== undefined && book.text_sim > 0 && <ScoreBadge score={book.text_sim} label="Text" />}
            {book.img_sim !== undefined && book.img_sim > 0 && <ScoreBadge score={book.img_sim} label="Img" />}
          </div>
        </div>
        <div className="flex gap-2 mt-2">
          <button
            onClick={() => onInteract(book, "click")}
            className="flex-1 py-1.5 rounded text-white/80 font-medium transition-all duration-200 hover:text-white"
            style={{ fontSize: 11, background: "linear-gradient(135deg, rgba(124,58,237,0.3), rgba(167,139,250,0.2))", border: "1px solid rgba(167,139,250,0.3)" }}
          >
            ✓ Interested
          </button>
          <button
            onClick={() => onInteract(book, "skip")}
            className="flex-1 py-1.5 rounded text-white/40 transition-all duration-200 hover:text-white/60"
            style={{ fontSize: 11, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}
          >
            ✕ Skip
          </button>
        </div>
        <div className="flex gap-2 mt-1.5 relative">
          <button
            onClick={handleAI}
            className="flex-1 py-1.5 rounded text-indigo-200 transition-all duration-200 hover:bg-indigo-500/20"
            style={{ fontSize: 11, background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.25)" }}
          >
            {showAI ? "✕ Close AI" : " Ask AI"}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onInteract(book, "cart"); }}
            className="flex-1 py-1.5 rounded text-emerald-200 transition-all duration-200 hover:bg-emerald-500/20 shadow-sm shadow-emerald-900/20"
            style={{ fontSize: 11, background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.4)" }}
          >
            Add to Cart
          </button>

          {/* AI Popover Tooltip */}
          {showAI && (
            <div
              className="absolute z-20 p-3 rounded-lg shadow-xl fade-in text-left"
              style={{
                top: "calc(100% + 8px)", left: 0, right: 0, minWidth: 260,
                background: "rgba(30,27,75,0.95)", border: "1px solid rgba(99,102,241,0.3)",
                backdropFilter: "blur(12px)"
              }}
            >
              <div className="flex flex-col gap-1 mb-3">
                <span className="text-white font-bold" style={{ fontSize: 14 }}>{book.title}</span>
                <span className="text-indigo-300" style={{ fontSize: 11 }}>by {displayAuthor}</span>
              </div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.85)", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
                {aiLoading ? <span className="shimmer text-indigo-200/50">Thinking...</span> :
                  aiText.split('\n').map((line, i) => (
                    <p key={i} className={line.startsWith('**') ? "mt-1.5" : ""}>
                      {line.split(/(\*\*.*?\*\*)/).map((part, j) =>
                        part.startsWith('**') && part.endsWith('**') ?
                          <strong key={j} className="text-white font-semibold">{part.slice(2, -2)}</strong> :
                          part
                      )}
                    </p>
                  ))
                }
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function RecommendCard({ book, onInteract, onAskAI, rank }) {
  const [showAI, setShowAI] = useState(false);
  const [aiText, setAiText] = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const handleAI = async (e) => {
    e.stopPropagation();
    if (showAI) {
      setShowAI(false);
      return;
    }
    setShowAI(true);
    if (!aiText) {
      setAiLoading(true);
      try {
        const text = await onAskAI(book);
        setAiText(text);
      } catch (err) {
        setAiText("Failed to query AI helper.");
      } finally {
        setAiLoading(false);
      }
    }
  };

  // Safe defaults
  const displayCoverColor = book.cover_color || "#1e1b4b";
  let displayAuthor = book.author || "Unknown Author";
  if (typeof displayAuthor === 'string' && displayAuthor.startsWith('{')) {
    try {
      const match = displayAuthor.match(/'name':\s*'([^']+)'/);
      if (match) displayAuthor = match[1];
    } catch(e) {}
  }

  return (
    <div
      className="flex gap-3 p-3 rounded-lg cursor-pointer group transition-all duration-300"
      style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
      onMouseEnter={e => { e.currentTarget.style.background = "rgba(52,211,153,0.06)"; e.currentTarget.style.borderColor = "rgba(52,211,153,0.2)"; }}
      onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.03)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; }}
    >
      <div className="flex flex-col items-center gap-2">
        <span className="text-white/20 font-mono font-bold" style={{ fontSize: 10 }}>#{rank + 1}</span>
        <BookCover color={displayCoverColor} title={book.title} size="sm" imageUrl={book.image_url} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white/85 font-medium" style={{ fontFamily: "'Playfair Display', Georgia, serif", fontSize: 12 }}>{book.title}</p>
        <p className="text-white/35 mt-0.5" style={{ fontSize: 10 }}>{displayAuthor}</p>
        <div className="flex items-center gap-2 mt-1.5">
          {book.layer && <LayerTag label={book.layer} />}
          <span className="text-white/30 font-mono" style={{ fontSize: 9 }}>match: {Math.min(99, Math.max(60, Math.floor((book.score || 0.5) * 80 + 15)))}%</span>
        </div>
        <div className="flex gap-1.5 mt-2">
          <button
            onClick={() => onInteract(book, "click")}
            className="flex-1 py-1 rounded text-white/70 transition-colors duration-150 hover:text-white"
            style={{ fontSize: 10, background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.25)" }}
          >
            ✓ Interested
          </button>
          <button
            onClick={() => onInteract(book, "skip")}
            className="flex-1 py-1 rounded text-white/30 transition-colors duration-150 hover:text-white/50"
            style={{ fontSize: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
          >
            ✗ Not for me
          </button>
        </div>
        <div className="flex gap-1.5 mt-1.5 relative">
          <button
            onClick={handleAI}
            className="flex-1 py-1 rounded text-indigo-200 transition-colors duration-150 hover:bg-indigo-500/20"
            style={{ fontSize: 10, background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.25)" }}
          >
            {showAI ? "✕ Close AI" : "Ask AI"}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onInteract(book, "cart"); }}
            className="flex-1 py-1 rounded text-emerald-200 transition-colors duration-150 hover:bg-emerald-500/20 shadow-sm shadow-emerald-900/20"
            style={{ fontSize: 10, background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.4)" }}
          >
            Add to Cart
          </button>

          {/* AI Popover Tooltip */}
          {showAI && (
            <div
              className="absolute z-20 p-2.5 rounded-lg shadow-xl fade-in text-left"
              style={{
                top: "calc(100% + 8px)", left: 0, right: -60, minWidth: 240,
                background: "rgba(30,27,75,0.95)", border: "1px solid rgba(99,102,241,0.3)",
                backdropFilter: "blur(12px)"
              }}
            >
              <div className="flex flex-col gap-1 mb-2">
                <span className="text-white font-bold" style={{ fontSize: 12 }}>{book.title}</span>
                <span className="text-indigo-300" style={{ fontSize: 10 }}>by {displayAuthor}</span>
              </div>
              <div style={{ fontSize: 10, color: "rgba(255,255,255,0.85)", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
                {aiLoading ? <span className="shimmer text-indigo-200/50">Thinking...</span> :
                  aiText.split('\n').map((line, i) => (
                    <p key={i} className={line.startsWith('**') ? "mt-1.5" : ""}>
                      {line.split(/(\*\*.*?\*\*)/).map((part, j) =>
                        part.startsWith('**') && part.endsWith('**') ?
                          <strong key={j} className="text-white font-semibold">{part.slice(2, -2)}</strong> :
                          part
                      )}
                    </p>
                  ))
                }
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  // ─── AUTH & SESSION ────────────────────────────────────────────────────────
  const [sessionId] = useState(() => {
    let id = localStorage.getItem("nba_session_id");
    if (!id) {
      id = "sess_" + Math.random().toString(36).substring(2, 15);
      localStorage.setItem("nba_session_id", id);
    }
    return id;
  });

  const [userId, setUserId] = useState("user_demo_01");
  const [isGuest, setIsGuest] = useState(false);

  const toggleUserMode = () => {
    if (isGuest) {
      setUserId("user_demo_01");
      setIsGuest(false);
    } else {
      setUserId(`guest_${sessionId.substring(5, 11)}`);
      setIsGuest(true);
    }
  };

  const [query, setQuery] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [recommendations, setRecommendations] = useState({ people_also_buy: [], you_might_like: [] });
  const [recTab, setRecTab] = useState("pab");
  const [interactions, setInteractions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingRecs, setIsLoadingRecs] = useState(false);
  const [rlStep, setRlStep] = useState(0);
  const [rlMetrics, setRlMetrics] = useState({ loss_history: [], buffer_size: 0, step: 0 });
  const [lastTrained, setLastTrained] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [useMock, setUseMock] = useState(false);
  const [activeTab, setActiveTab] = useState("search");
  const [isDark, setIsDark] = useState(true);
  const [activeRightTab, setActiveRightTab] = useState("profile");
  const [cart, setCart] = useState([]);

  const fileInputRef = useRef();
  const dropRef = useRef();

  const addToast = useCallback((msg, type = "info") => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3000);
  }, []);

  const handleSearch = async () => {
    if (!query.trim() && !imageFile) return;
    setIsSearching(true);
    try {
      if (useMock) {
        await new Promise(r => setTimeout(r, 800));
        setSearchResults(MOCK_BOOKS);
        addToast("Mock search: BLaIR + CLIP fusion complete", "success");
      } else {
        let imgB64 = null;
        if (imageFile) {
          imgB64 = await new Promise(res => {
            const r = new FileReader();
            r.onload = () => res(r.result.split(",")[1]);
            r.readAsDataURL(imageFile);
          });
        }
        const data = await apiSearch(query, imgB64, sessionId);
        setSearchResults(data.results || []);
        addToast(`Found ${data.results?.length} results`, "success");
      }
    } catch (e) {
      console.error("Search error:", e);
      addToast("Backend unreachable — enable mock mode", "error");
    } finally {
      setIsSearching(false);
    }
  };

  const loadRecommendations = async () => {
    setIsLoadingRecs(true);
    try {
      if (useMock) {
        await new Promise(r => setTimeout(r, 600));
        setRecommendations(MOCK_RECS);
        addToast("Mock recommendations loaded", "success");
      } else {
        const data = await apiRecommend(userId, sessionId);
        setRecommendations(data || { people_also_buy: [], you_might_like: [] });
        addToast("Personalized recommendations loaded", "success");
        try {
          const metrics = await apiRlMetrics(userId);
          setRlMetrics(metrics);
        } catch (e) {}
      }
    } catch (e) {
      addToast("Backend unreachable — enable mock mode", "error");
    } finally {
      setIsLoadingRecs(false);
    }
  };

  const handleInteract = async (book, action) => {
    const interaction = { ...book, action, ts: Date.now() };
    setInteractions(prev => [interaction, ...prev]);
    setRlStep(s => s + 1);

    if (action === "cart") {
      addToast(`"${book.title}" — Added to Cart`, "success");
      setCart(prev => [...prev, book]);
    } else if (action === "click") {
      addToast(`Clicked "${book.title}" — profile updated`, "success");
    } else {
      addToast(`✕ Skipped "${book.title}" — RL penalized`, "info");
    }

    try {
      if (!useMock) {
        await apiInteract(userId, book.id, action, sessionId);
        try {
          const metrics = await apiRlMetrics(userId);
          setRlMetrics(metrics);
        } catch (e) {}
      } else {
         setRlMetrics(p => ({
            loss_history: [...p.loss_history, Math.max(0.1, (p.loss_history[p.loss_history.length - 1] || 0.8) - 0.05 + (Math.random()*0.02 - 0.01))].slice(-100),
            buffer_size: Math.min(2000, p.buffer_size + 1),
            step: p.step + 1
         }));
      }
      setLastTrained(new Date());
      if ((rlStep + 1) % 3 === 0) {
        setTimeout(() => loadRecommendations(), 800);
      }
    } catch (e) {
      // silent fail for mock
    }
  };

  const handleImageDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
      addToast("Image loaded — CLIP encoder ready", "info");
    }
  };

  useEffect(() => {
    // Initial load sync
    if (!useMock) {
      loadRecommendations();
    }
  }, [useMock]);

  useEffect(() => {
    loadRecommendations();
  }, []);

  const handleAskAI = async (book) => {
    if (useMock) {
      await new Promise(r => setTimeout(r, 1500));
      return "This is a fantastic book! The author masterfully weaves a gripping narrative that keeps you hooked from page one. Highly recommended!";
    } else {
      const data = await apiAskLLM(book.title, book.author, "Why should I read this book? Give me a short 2-sentence pitch.");
      return data.response;
    }
  };

  const ctr = interactions.length > 0
    ? (interactions.filter(i => i.action === "click" || i.action === "cart").length / interactions.length * 100).toFixed(1)
    : "—";

  return (
    <div className={`min-h-screen text-white ${!isDark ? "theme-light" : ""}`} style={{
      background: "#080b14",
      fontFamily: "'DM Sans', system-ui, sans-serif",
    }}>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=Playfair+Display:wght@400;600;700&family=DM+Mono:wght@300;400&display=swap');
        * { box-sizing: border-box; }
        html, body, #root { height: 100%; overflow: hidden; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 2px; }
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes shimmer { 0%,100% { opacity: 0.4; } 50% { opacity: 0.8; } }
        @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.4); } 50% { box-shadow: 0 0 0 4px rgba(52,211,153,0); } }
        .fade-in { animation: fadeSlideIn 0.3s ease forwards; }
        .shimmer { animation: shimmer 1.5s ease infinite; }
        button { outline: none; }
        input { outline: none; }
        .theme-light {
          filter: invert(1) hue-rotate(180deg);
        }
        .theme-light img, .theme-light .book-cover-uninvert {
          filter: invert(1) hue-rotate(180deg);
        }
      `}</style>

      {/* Noise texture overlay */}
      <div className="fixed inset-0 pointer-events-none" style={{
        backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
        opacity: 0.018,
        zIndex: 0,
      }} />

      {/* Ambient glows */}
      <div className="fixed pointer-events-none" style={{
        width: 600, height: 600, borderRadius: "50%",
        background: "radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%)",
        top: -200, left: -100, zIndex: 0,
      }} />
      <div className="fixed pointer-events-none" style={{
        width: 400, height: 400, borderRadius: "50%",
        background: "radial-gradient(circle, rgba(52,211,153,0.04) 0%, transparent 70%)",
        bottom: -100, right: 200, zIndex: 0,
      }} />

      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2" style={{ maxWidth: 300 }}>
        {toasts.map(t => (
          <div key={t.id} className="fade-in px-3 py-2 rounded-lg backdrop-blur-sm" style={{
            background: t.type === "success" ? "rgba(52,211,153,0.12)" : t.type === "error" ? "rgba(239,68,68,0.12)" : "rgba(167,139,250,0.12)",
            border: `1px solid ${t.type === "success" ? "rgba(52,211,153,0.25)" : t.type === "error" ? "rgba(239,68,68,0.25)" : "rgba(167,139,250,0.25)"}`,
            color: t.type === "success" ? "#6ee7b7" : t.type === "error" ? "#fca5a5" : "#c4b5fd",
            fontSize: 11,
            boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
          }}>
            {t.msg}
          </div>
        ))}
      </div>

      {/* ── HEADER ── */}
      <header className="relative z-10 flex items-center justify-between px-6 py-3.5" style={{
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(8,11,20,0.8)",
        backdropFilter: "blur(12px)",
      }}>
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0" style={{
            background: "linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%)",
            boxShadow: "0 0 20px rgba(124,58,237,0.4), inset 0 1px 0 rgba(255,255,255,0.15)",
          }}>
            <span style={{ fontSize: 16 }}></span>
          </div>
          <div>
            <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 17, fontWeight: 700, lineHeight: 1.1, letterSpacing: "0.01em" }}>
              NBA<span style={{ color: "#a78bfa" }}>sys</span>
            </h1>
            <p style={{ fontSize: 9, color: "rgba(255,255,255,0.28)", marginTop: 1, fontFamily: "'DM Mono', monospace", letterSpacing: "0.05em" }}>
              DATN · MULTIMODAL REC ENGINE
            </p>
          </div>
        </div>

        {/* Controls & Stats */}
        <div className="flex items-center gap-6">
          {/* User Mode Toggle */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={toggleUserMode}
              className="px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-2"
              style={{
                background: isGuest ? "rgba(167,139,250,0.1)" : "rgba(52,211,153,0.1)",
                border: `1px solid ${isGuest ? "rgba(167,139,250,0.3)" : "rgba(52,211,153,0.3)"}`,
                color: isGuest ? "#a78bfa" : "#6ee7b7",
                fontSize: 12,
                fontWeight: 600
              }}
            >
              <span style={{ fontSize: 14 }}>{isGuest ? "👤" : "👨‍💻"}</span>
              {isGuest ? "Guest Mode" : "Demo User"}
            </button>
          </div>

          <div style={{ width: 1, height: 28, background: "rgba(255,255,255,0.08)" }} />

          {/* Cart, Theme & Mock/Live toggle */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsDark(!isDark)}
              className="px-3 py-1.5 rounded-lg transition-all duration-200"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", fontSize: 13 }}
            >
              {isDark ? "Light" : "Dark"}
            </button>
            <button
              className="px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-2"
              style={{ background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.3)", color: "#10b981", fontSize: 13, fontWeight: 600 }}
            >
              Cart <span style={{ background: "#10b981", color: "#000", padding: "0 6px", borderRadius: 10, fontSize: 10 }}>{cart.length}</span>
            </button>
          </div>
          
          <div style={{ width: 1, height: 28, background: "rgba(255,255,255,0.08)" }} />

          {/* Mock/Live */}
          <div className="flex items-center gap-2.5">
            <span style={{ fontSize: 10, color: useMock ? "rgba(167,139,250,0.9)" : "rgba(255,255,255,0.25)", fontFamily: "'DM Mono', monospace", fontWeight: 400, letterSpacing: "0.04em" }}>MOCK</span>
            <button
              onClick={() => { setUseMock(m => !m); addToast(useMock ? "Switched to Live API mode" : "Switched to Mock mode", "info"); }}
              className="relative rounded-full transition-all duration-300 flex-shrink-0"
              style={{
                width: 36, height: 18,
                background: useMock ? "rgba(124,58,237,0.5)" : "rgba(52,211,153,0.5)",
                border: `1px solid ${useMock ? "rgba(167,139,250,0.4)" : "rgba(52,211,153,0.4)"}`,
                boxShadow: useMock ? "0 0 12px rgba(124,58,237,0.3)" : "0 0 12px rgba(52,211,153,0.3)",
                cursor: "pointer",
              }}
            >
              <div
                className="absolute rounded-full bg-white transition-all duration-300"
                style={{
                  width: 12, height: 12, top: 2,
                  left: useMock ? 2 : 20,
                  boxShadow: "0 1px 4px rgba(0,0,0,0.5)",
                }}
              />
            </button>
            <span style={{ fontSize: 10, color: !useMock ? "rgba(52,211,153,0.9)" : "rgba(255,255,255,0.25)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.04em" }}>LIVE</span>
          </div>

          {/* Separator */}
          <div style={{ width: 1, height: 28, background: "rgba(255,255,255,0.08)" }} />

          {/* Stats */}
          <div className="flex gap-5">
            {[
              { label: "RL Steps", value: rlStep, color: "#a78bfa" },
              { label: "CTR", value: `${ctr}%`, color: "#6ee7b7" },
              { label: "Interactions", value: interactions.length, color: "#60a5fa" },
            ].map(s => (
              <div key={s.label} className="text-center">
                <div className="font-mono font-semibold" style={{ fontSize: 15, color: s.value === 0 || s.value === "—%" ? "rgba(255,255,255,0.3)" : s.color }}>{s.value}</div>
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.04em" }}>{s.label.toUpperCase()}</div>
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* ── MAIN LAYOUT ── */}
      <div className="relative z-10 flex" style={{ height: "calc(100vh - 61px)" }}>

        {/* ── LEFT PANEL (42%) — Search / Recs tabs ── */}
        <div className="flex flex-col flex-shrink-0" style={{ width: "42%", borderRight: "1px solid rgba(255,255,255,0.06)" }}>

          {/* Tab bar */}
          <div className="flex px-4 pt-3 gap-1" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
            {[
              ["search", "Active Search", "Mode 1"],
              ["recs", "Recommendations", "Mode 2"],
            ].map(([tab, label, mode]) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="flex-1 py-2.5 text-center transition-all duration-200 rounded-t relative"
                style={{
                  fontSize: 11,
                  fontWeight: activeTab === tab ? 500 : 400,
                  color: activeTab === tab ? "#c4b5fd" : "rgba(255,255,255,0.28)",
                  background: activeTab === tab ? "rgba(124,58,237,0.08)" : "transparent",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                {label}
                <span style={{ fontSize: 9, color: "rgba(255,255,255,0.18)", marginLeft: 4 }}>{mode}</span>
                {activeTab === tab && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t" style={{ background: "linear-gradient(to right, #7c3aed, #a78bfa)" }} />
                )}
              </button>
            ))}
          </div>

          {/* ── SEARCH TAB ── */}
          {activeTab === "search" ? (
            <div className="flex flex-col flex-1 overflow-hidden px-4 pt-4 gap-3">

              {/* Search controls */}
              <div className="space-y-2">
                {/* Text input */}
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      onKeyDown={e => e.key === "Enter" && handleSearch()}
                      placeholder='e.g. "Dark Fantasy with complex magic systems"'
                      className="w-full px-3 py-2.5 rounded-xl text-white/80 placeholder-white/20 transition-all duration-200"
                      style={{
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid rgba(255,255,255,0.09)",
                        fontSize: 12,
                        fontFamily: "'DM Sans', sans-serif",
                        color: "rgba(255,255,255,0.85)",
                      }}
                      onFocus={e => e.target.style.borderColor = "rgba(167,139,250,0.5)"}
                      onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.09)"}
                    />
                  </div>
                  <button
                    onClick={handleSearch}
                    disabled={isSearching}
                    className="px-5 py-2.5 rounded-xl font-medium transition-all duration-200"
                    style={{
                      fontSize: 12,
                      background: isSearching
                        ? "rgba(167,139,250,0.15)"
                        : "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)",
                      border: "1px solid rgba(167,139,250,0.3)",
                      boxShadow: isSearching ? "none" : "0 0 20px rgba(124,58,237,0.3)",
                      cursor: isSearching ? "not-allowed" : "pointer",
                      color: "white",
                      flexShrink: 0,
                    }}
                  >
                    {isSearching ? <span className="shimmer">…</span> : "Search"}
                  </button>
                </div>

                {/* Image drop zone */}
                <div
                  ref={dropRef}
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleImageDrop}
                  onDragOver={e => e.preventDefault()}
                  className="relative rounded-xl flex items-center gap-3 cursor-pointer transition-all duration-200"
                  style={{
                    padding: imagePreview ? "8px 12px" : "10px 14px",
                    background: imagePreview ? "rgba(167,139,250,0.07)" : "rgba(255,255,255,0.02)",
                    border: `1px dashed ${imagePreview ? "rgba(167,139,250,0.4)" : "rgba(255,255,255,0.09)"}`,
                  }}
                  onMouseEnter={e => { if (!imagePreview) { e.currentTarget.style.borderColor = "rgba(167,139,250,0.3)"; e.currentTarget.style.background = "rgba(167,139,250,0.04)"; } }}
                  onMouseLeave={e => { if (!imagePreview) { e.currentTarget.style.borderColor = "rgba(255,255,255,0.09)"; e.currentTarget.style.background = "rgba(255,255,255,0.02)"; } }}
                >
                  {imagePreview ? (
                    <>
                      <img src={imagePreview} className="w-11 h-11 rounded-lg object-cover flex-shrink-0" style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.4)" }} alt="query" />
                      <div className="flex-1 min-w-0">
                        <p style={{ fontSize: 11, color: "rgba(255,255,255,0.6)", fontWeight: 500 }}>Image query loaded</p>
                        <p style={{ fontSize: 9, color: "rgba(255,255,255,0.3)" }}>CLIP will encode this for visual similarity</p>
                      </div>
                      <button
                        onClick={e => { e.stopPropagation(); setImageFile(null); setImagePreview(null); }}
                        className="flex-shrink-0 transition-colors duration-150"
                        style={{ fontSize: 14, color: "rgba(255,255,255,0.3)", background: "none", border: "none", cursor: "pointer", padding: 4 }}
                        onMouseEnter={e => e.currentTarget.style.color = "rgba(255,255,255,0.7)"}
                        onMouseLeave={e => e.currentTarget.style.color = "rgba(255,255,255,0.3)"}
                      >✕</button>
                    </>
                  ) : (
                    <>
                      <span style={{ fontSize: 20, opacity: 0.35, flexShrink: 0 }}></span>
                      <div>
                        <p style={{ fontSize: 11, color: "rgba(255,255,255,0.32)" }}>Drop a cover image here</p>
                        <p style={{ fontSize: 9, color: "rgba(255,255,255,0.18)" }}>CLIP image encoder · visual similarity search</p>
                      </div>
                    </>
                  )}
                  <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleImageDrop} />
                </div>
              </div>

              {/* Encoder legend */}
              <div className="flex items-center gap-3">
                {[["BLaIR", "#7c3aed", "text semantics"], ["CLIP", "#2563eb", "visual features"], ["RRF Fusion", "#059669", "final score"]].map(([name, color, desc]) => (
                  <div key={name} className="flex items-center gap-1.5">
                    <div className="rounded-full flex-shrink-0" style={{ width: 6, height: 6, background: color, boxShadow: `0 0 6px ${color}` }} />
                    <span style={{ fontSize: 9, color: "rgba(255,255,255,0.38)" }}>
                      <span style={{ color: "rgba(255,255,255,0.65)", fontWeight: 500 }}>{name}</span> · {desc}
                    </span>
                  </div>
                ))}
              </div>

              {/* Results list */}
              <div className="flex-1 overflow-y-auto space-y-2 pr-1" style={{ marginRight: -4 }}>
                {isSearching ? (
                  <div className="flex flex-col items-center justify-center h-40 gap-3">
                    <div className="flex gap-1.5">
                      {[0, 1, 2].map(i => (
                        <div key={i} className="rounded-full" style={{ width: 8, height: 8, background: "#7c3aed", animation: `shimmer 1s ease ${i * 0.2}s infinite` }} />
                      ))}
                    </div>
                    <p style={{ fontSize: 11, color: "rgba(255,255,255,0.28)" }}>Encoding query → FAISS search…</p>
                  </div>
                ) : searchResults.length > 0 ? (
                  searchResults.map((book, i) => (
                    <SearchResultCard key={i} book={book} onInteract={handleInteract} onAskAI={handleAskAI} />
                  ))
                ) : (
                  <div className="flex flex-col items-center justify-center h-48 text-center gap-3">
                    <div style={{ fontSize: 40, opacity: 0.15 }}></div>
                    <div>
                      <p style={{ fontSize: 13, color: "rgba(255,255,255,0.22)", fontWeight: 500 }}>Search for a book to begin</p>
                      <p style={{ fontSize: 10, color: "rgba(255,255,255,0.12)", marginTop: 4 }}>BLaIR + CLIP · 3M book catalog</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

          ) : (
            /* ── RECOMMENDATIONS TAB ── */
            <div className="flex flex-col flex-1 overflow-hidden px-4 pt-4 gap-3">
              <div className="flex items-center justify-between">
                <div>
                  <p style={{ fontSize: 12, color: "rgba(255,255,255,0.6)", fontWeight: 500 }}>Personalized For You</p>
                  <p style={{ fontSize: 10, color: "rgba(255,255,255,0.22)", marginTop: 2 }}>Retrieval + RL-DQN Multi-mode</p>
                </div>
                <button
                  onClick={loadRecommendations}
                  disabled={isLoadingRecs}
                  className="transition-all duration-200"
                  style={{
                    fontSize: 11, padding: "6px 14px",
                    background: "rgba(52,211,153,0.08)",
                    border: "1px solid rgba(52,211,153,0.22)",
                    borderRadius: 10,
                    color: isLoadingRecs ? "rgba(255,255,255,0.3)" : "rgba(52,211,153,0.9)",
                    cursor: isLoadingRecs ? "not-allowed" : "pointer",
                  }}
                  onMouseEnter={e => !isLoadingRecs && (e.currentTarget.style.background = "rgba(52,211,153,0.14)")}
                  onMouseLeave={e => e.currentTarget.style.background = "rgba(52,211,153,0.08)"}
                >
                  {isLoadingRecs ? <span className="shimmer">Fetching…</span> : "↺ Refresh"}
                </button>
              </div>

              {/* Sub-tab Toggle */}
              <div className="flex rounded-lg p-1" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                {[
                  { id: "pab", label: "People Also Buy", desc: "Cleora + BLaIR/CLIP", color: "#60a5fa" },
                  { id: "yml", label: "You Might Like", desc: "RL-DQN Personalization", color: "#a78bfa" }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setRecTab(tab.id)}
                    className="flex-1 flex flex-col items-center py-1.5 rounded-md transition-all duration-200"
                    style={{
                      background: recTab === tab.id ? `${tab.color}15` : "transparent",
                      border: `1px solid ${recTab === tab.id ? `${tab.color}40` : "transparent"}`,
                    }}
                  >
                    <span style={{ fontSize: 11, fontWeight: recTab === tab.id ? 600 : 400, color: recTab === tab.id ? tab.color : "rgba(255,255,255,0.4)" }}>{tab.label}</span>
                    <span style={{ fontSize: 9, color: "rgba(255,255,255,0.2)" }}>{tab.desc}</span>
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                {isLoadingRecs ? (
                  <div className="flex flex-col items-center justify-center h-40 gap-3">
                    <div className="flex gap-1.5">
                      {[0, 1, 2].map(i => (
                        <div key={i} className="rounded-full" style={{ width: 8, height: 8, background: "#34d399", animation: `shimmer 1s ease ${i * 0.2}s infinite` }} />
                      ))}
                    </div>
                    <p style={{ fontSize: 11, color: "rgba(255,255,255,0.28)" }}>Generating recommendations…</p>
                  </div>
                ) : (
                  (recTab === "pab" ? recommendations.people_also_buy : recommendations.you_might_like)?.map((book, i) => (
                    <RecommendCard key={i} book={book} rank={i} onInteract={handleInteract} onAskAI={handleAskAI} />
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL — Profile + Activity ── */}
        <div className="flex flex-col flex-1 overflow-hidden min-w-0">
          <div className="flex px-4 pt-3 gap-1" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
            {[
              ["profile", "User Profile"],
              ["history", "Activity History"],
            ].map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setActiveRightTab(tab)}
                className="flex-1 py-2.5 text-center transition-all duration-200 rounded-t relative"
                style={{
                  fontSize: 11, fontWeight: activeRightTab === tab ? 500 : 400,
                  color: activeRightTab === tab ? "#c4b5fd" : "rgba(255,255,255,0.28)",
                  background: activeRightTab === tab ? "rgba(124,58,237,0.08)" : "transparent",
                  border: "none", cursor: "pointer",
                }}
              >
                {label}
                {activeRightTab === tab && <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t" style={{ background: "linear-gradient(to right, #7c3aed, #a78bfa)" }} />}
              </button>
            ))}
          </div>
          {activeRightTab === "profile" ? (
            <div className="flex flex-col flex-1 overflow-y-auto">

          {/* User Profile State */}
          <div className="px-5 py-4 flex-shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.8)" }}>User Profile State</h2>
                <p style={{ fontSize: 10, color: "rgba(255,255,255,0.22)", marginTop: 2 }}>Aggregated embedding · temporal decay λ=0.1</p>
              </div>
              <div className="text-right">
                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", fontFamily: "'DM Mono', monospace" }}>{userId}</div>
                {lastTrained && (
                  <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", marginTop: 2 }}>
                    last update: {lastTrained.toLocaleTimeString()}
                  </div>
                )}
              </div>
            </div>
            <ProfileRadar interactions={interactions} />
          </div>

          {/* RL Training Feed */}
          <div className="px-5 py-3 flex-shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="flex items-center justify-between mb-2.5">
              <h3 style={{ fontSize: 11, color: "rgba(255,255,255,0.45)", fontWeight: 500 }}>RL Training Feed</h3>
              <div className="flex items-center gap-2">
                <div style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: rlStep > 0 ? "#34d399" : "rgba(255,255,255,0.15)",
                  animation: rlStep > 0 ? "pulse 2s ease infinite" : "none",
                }} />
                <span style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", fontFamily: "'DM Mono', monospace" }}>{rlStep} steps</span>
              </div>
            </div>
            <div className="space-y-1.5" style={{ maxHeight: 88, overflowY: "auto" }}>
              {interactions.slice(0, 6).map((item, i) => (
                <div key={i} className="flex items-center gap-2 fade-in">
                  <span style={{ fontSize: 9, color: (item.action === "click" || item.action === "cart") ? "#34d399" : "#f87171", fontFamily: "'DM Mono', monospace", minWidth: 64 }}>
                    {(item.action === "click" || item.action === "cart") ? "▲ +reward" : "▼ −reward"}
                  </span>
                  <span className="flex-1 truncate" style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace" }}>
                    {item.title}
                  </span>
                  <span style={{ fontSize: 8, color: "rgba(255,255,255,0.14)", fontFamily: "'DM Mono', monospace", flexShrink: 0 }}>
                    step {rlStep - i}
                  </span>
                </div>
              ))}
              {interactions.length === 0 && (
                <p style={{ fontSize: 10, color: "rgba(255,255,255,0.18)" }}>No interactions yet — click or skip items to train</p>
              )}
            </div>
          </div>

          {/* RL Loss Metric */}
          <div className="px-5 py-3 flex-shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="flex items-center justify-between mb-2">
              <div>
                <h3 style={{ fontSize: 11, color: "rgba(255,255,255,0.45)", fontWeight: 500 }}>DQN Training Loss</h3>
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", marginTop: 1, fontFamily: "'DM Mono', monospace" }}>{rlMetrics.buffer_size}/2000 transitions</div>
              </div>
              <div style={{ fontSize: 13, color: "#a78bfa", fontFamily: "'DM Mono', monospace", fontWeight: 600 }}>
                {rlMetrics.loss_history.length > 0 ? rlMetrics.loss_history[rlMetrics.loss_history.length - 1].toFixed(4) : "0.0000"}
              </div>
            </div>
            {/* Sparkline */}
            {rlMetrics.loss_history.length > 1 ? (
              <div className="h-8 w-full mt-2 relative flex items-end overflow-hidden" style={{ borderBottom: "1px dotted rgba(255,255,255,0.1)" }}>
                <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none" viewBox={`0 0 ${Math.max(1, rlMetrics.loss_history.length - 1)} 1`}>
                  <polyline
                    fill="none"
                    stroke="rgba(167,139,250,0.8)"
                    strokeWidth="0.05"
                    vectorEffect="non-scaling-stroke"
                    points={rlMetrics.loss_history.map((val, i) => {
                       const min = Math.min(...rlMetrics.loss_history);
                       const max = Math.max(...rlMetrics.loss_history);
                       const range = max - min || 1;
                       const norm = 1 - (val - min) / range;
                       return `${i},${norm}`;
                    }).join(' ')}
                  />
                </svg>
              </div>
            ) : (
              <div className="h-8 w-full mt-2 flex items-center justify-center" style={{ background: "rgba(255,255,255,0.02)", borderRadius: 4 }}>
                <span style={{ fontSize: 9, color: "rgba(255,255,255,0.1)" }}>Interact to stream loss data</span>
              </div>
            )}
          </div>

          </div>
          ) : (
            <div className="flex-col flex-1 overflow-y-auto">
              {/* Interaction History */}
          <div className="flex-1 overflow-y-auto px-5 py-3">
            <h3 style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", fontWeight: 500, marginBottom: 12 }}>Interaction History</h3>
            {interactions.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 text-center gap-3">
                <div>
                  <p style={{ fontSize: 11, color: "rgba(255,255,255,0.18)" }}>Interact with search results or<br />recommendations to train the DQN</p>
                  <p style={{ fontSize: 9, color: "rgba(255,255,255,0.1)", marginTop: 6 }}>Profile fingerprint updates in real-time</p>
                </div>
              </div>
            ) : (
              <div className="space-y-1">
                {interactions.map((item, i) => (
                  <div key={i} className="flex items-center gap-3 py-2 fade-in" style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <BookCover color={item.cover_color} title={item.title} size="sm" imageUrl={item.image_url} />
                    <div className="flex-1 min-w-0">
                      <p className="truncate" style={{ fontFamily: "'Playfair Display', serif", fontSize: 11, color: "rgba(255,255,255,0.65)" }}>{item.title}</p>
                      <p style={{ fontSize: 9, color: "rgba(255,255,255,0.25)", marginTop: 1 }}>{item.author}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      <span style={{
                        fontSize: 9, padding: "2px 8px", borderRadius: 99,
                        background: (item.action === "click" || item.action === "cart") ? "rgba(52,211,153,0.1)" : "rgba(248,113,113,0.1)",
                        color: (item.action === "click" || item.action === "cart") ? "#6ee7b7" : "#fca5a5",
                        border: `1px solid ${(item.action === "click" || item.action === "cart") ? "rgba(52,211,153,0.22)" : "rgba(248,113,113,0.22)"}`,
                        fontWeight: 500,
                      }}>
                        {item.action}
                      </span>
                      <span style={{ fontSize: 8, color: "rgba(255,255,255,0.12)", fontFamily: "'DM Mono', monospace" }}>
                        {new Date(item.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}