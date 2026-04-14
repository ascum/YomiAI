import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { api } from "./services/api";
import { BookCover } from "./components/ui/BookCover";
import { ProfileRadar } from "./components/features/profile/ProfileRadar";
import { SearchResultCard } from "./components/features/search/SearchResultCard";
import { RecommendCard } from "./components/features/recs/RecommendCard";
import { SkeletonCard } from "./components/ui/SkeletonCard";
import { LoginPage } from "./components/LoginPage";

// ─── Mock data ────────────────────────────────────────────────────────────────
const MOCK_BOOKS = [
  { id:"b001", title:"The Name of the Wind",       author:"Patrick Rothfuss",        genre:"Dark Fantasy",        score:0.94, cover_color:"#1a0a2e", text_sim:0.91, img_sim:0.88 },
  { id:"b002", title:"Mistborn: The Final Empire", author:"Brandon Sanderson",       genre:"Epic Fantasy",        score:0.89, cover_color:"#0d1b2a", text_sim:0.85, img_sim:0.82 },
  { id:"b003", title:"The Way of Kings",           author:"Brandon Sanderson",       genre:"High Fantasy",        score:0.87, cover_color:"#162040", text_sim:0.83, img_sim:0.79 },
  { id:"b004", title:"A Shadow in the Ember",      author:"Jennifer L. Armentrout", genre:"Dark Romance Fantasy", score:0.82, cover_color:"#2d0a0a", text_sim:0.78, img_sim:0.81 },
  { id:"b005", title:"Blood and Ash",              author:"Jennifer L. Armentrout", genre:"Dark Fantasy",        score:0.80, cover_color:"#1a0000", text_sim:0.76, img_sim:0.83 },
];
const MOCK_RECS = {
  people_also_buy: [
    { id:"r001", title:"The Lies of Locke Lamora", author:"Scott Lynch",      score:0.91, cover_color:"#0a1a2d", layer:"Cleora + BGE-M3" },
    { id:"r002", title:"Assassin's Apprentice",    author:"Robin Hobb",       score:0.88, cover_color:"#0d2010", layer:"Cleora + CLIP" },
  ],
  you_might_like: [
    { id:"r003", title:"The Black Prism",  author:"Brent Weeks",      score:0.86, cover_color:"#1a1000", layer:"DIF-SASRec" },
    { id:"r004", title:"Kushiel's Dart",   author:"Jacqueline Carey", score:0.84, cover_color:"#100808", layer:"DIF-SASRec" },
  ],
};

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  // ── Theme (initialised first — LoginPage needs it) ──────────────────────────
  const [isDark, setIsDark] = useState(false);
  useEffect(() => { document.documentElement.classList.toggle("dark", isDark); }, [isDark]);

  // ── Session & auth ──────────────────────────────────────────────────────────
  const [sessionId] = useState(() => {
    let id = localStorage.getItem("yomiai_session_id");
    if (!id) { id = "sess_" + Math.random().toString(36).substring(2, 15); localStorage.setItem("yomiai_session_id", id); }
    return id;
  });

  // userId: read from localStorage on mount so returning users skip the gate
  const [userId, setUserId] = useState(() => localStorage.getItem("yomiai_user_id") || null);
  const [isGuest, setIsGuest] = useState(() => (localStorage.getItem("yomiai_user_id") || "").startsWith("guest_"));

  const handleLogin = (uid) => {
    localStorage.setItem("yomiai_user_id", uid);
    setUserId(uid);
    setIsGuest(uid.startsWith("guest_"));
  };

  const handleLogout = () => {
    localStorage.removeItem("yomiai_user_id");
    setUserId(null);
  };

  // ── Search ──────────────────────────────────────────────────────────────────
  const [query, setQuery]               = useState("");
  const [imageFile, setImageFile]       = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching]   = useState(false);
  const [searchError, setSearchError]   = useState(null);

  // ── Recommendations ─────────────────────────────────────────────────────────
  const [recommendations, setRecommendations] = useState({ people_also_buy: [], you_might_like: [] });
  const [recTab, setRecTab]             = useState("pab");
  const [isLoadingRecs, setLoadingRecs] = useState(false);
  const [recsError, setRecsError]       = useState(null);
  const [lastRecsRefresh, setLastRecsRefresh] = useState(null);

  // ── RL / profile ────────────────────────────────────────────────────────────
  const [interactions, setInteractions] = useState([]);
  const [rlStep, setRlStep]             = useState(0);
  const [rlMetrics, setRlMetrics]       = useState({ loss_history: [], step: 0, arch: "" });
  const [lastTrained, setLastTrained]   = useState(null);
  const [profileStats, setProfileStats] = useState({ recent_items: [] });

  // ── UI state ────────────────────────────────────────────────────────────────
  const [toasts, setToasts]           = useState([]);
  const [useMock, setUseMock]         = useState(false);
  const [activeTab, setActiveTab]     = useState("search");
  const [activeRight, setActiveRight] = useState("profile");
  const [cart, setCart]               = useState([]);

  const fileRef = useRef();
  const dropRef = useRef();

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const toast = useCallback((msg, type = "info") => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3000);
  }, []);

  const loadProfile = async () => {
    if (useMock) return;
    try { const d = await api.profile(userId); if (d) { setProfileStats(d); setInteractions(d.recent_items || []); } } catch (_) {}
  };

  const loadRecs = useCallback(async () => {
    setLoadingRecs(true);
    setRecsError(null);
    try {
      if (useMock) {
        await new Promise(r => setTimeout(r, 600));
        setRecommendations(MOCK_RECS);
      } else {
        const d = await api.recommend(userId, sessionId);
        setRecommendations(d || { people_also_buy: [], you_might_like: [] });
        try { const m = await api.rlMetrics(userId); setRlMetrics(m); } catch (_) {}
      }
      setLastRecsRefresh(new Date());
    } catch (_) {
      setRecsError("Could not load recommendations.");
      toast("Backend unreachable — enable mock mode", "error");
    } finally { setLoadingRecs(false); }
  }, [userId, useMock, sessionId, toast]);

  // Single effect — fires on mount and whenever userId/useMock change
  useEffect(() => { loadRecs(); loadProfile(); }, [userId, useMock]);

  // ── Search handler ──────────────────────────────────────────────────────────
  const handleSearch = async () => {
    if (!query.trim() && !imageFile) return;
    setIsSearching(true);
    setSearchError(null);
    try {
      if (useMock) {
        await new Promise(r => setTimeout(r, 800));
        setSearchResults(MOCK_BOOKS);
        toast("Mock search: BGE-M3 + CLIP fusion complete", "success");
      } else {
        let imgB64 = null;
        if (imageFile) imgB64 = await new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result.split(",")[1]); r.readAsDataURL(imageFile); });
        const d = await api.search(query, imgB64, sessionId);
        setSearchResults(d.results || []);
        toast(`Found ${d.results?.length ?? 0} results`, "success");
      }
    } catch (_) {
      setSearchError("Search failed. Is the backend running?");
      toast("Backend unreachable — enable mock mode", "error");
    } finally { setIsSearching(false); }
  };

  // ── Interact handler ────────────────────────────────────────────────────────
  const handleInteract = async (book, action) => {
    setInteractions(p => [{ ...book, action, ts: Date.now() }, ...p]);
    setRlStep(s => s + 1);
    if (action === "cart")       { toast(`"${book.title}" — Added to Cart`, "success"); setCart(p => [...p, book]); }
    else if (action === "click") { toast(`Clicked "${book.title}" — profile updated`, "success"); }
    else                         { toast(`✕ Skipped "${book.title}" — RL penalized`, "info"); }
    try {
      if (!useMock) {
        await api.interact(userId, book.id, action, sessionId);
        try { const m = await api.rlMetrics(userId); setRlMetrics(m); } catch (_) {}
      } else {
        setRlMetrics(p => ({
          loss_history: [...p.loss_history, Math.max(0.1, (p.loss_history.at(-1) || 0.8) - 0.05 + (Math.random() * 0.02 - 0.01))].slice(-100),
          step: p.step + 1,
          arch: "DIF-SASRec",
        }));
      }
      setLastTrained(new Date());
      if ((rlStep + 1) % 3 === 0) setTimeout(loadRecs, 800);
    } catch (_) {}
  };

  const handleImageDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0];
    if (file?.type.startsWith("image/")) {
      setImageFile(file);
      const r = new FileReader(); r.onload = () => setImagePreview(r.result); r.readAsDataURL(file);
      toast("Image loaded — CLIP encoder ready", "info");
    }
  };

  const handleAskAI = async (book) => {
    if (useMock) { await new Promise(r => setTimeout(r, 1500)); return "A fantastic book the author masterfully weaves a gripping narrative. Highly recommended!"; }
    const d = await api.askLLM(book.title, book.author, "Why should I read this? Give a short 2-sentence pitch.");
    return d.response;
  };

  const handleAskAIStream = (book) => {
    if (useMock) {
      // Return a generator that mimics a stream for mock mode
      return (async function* () {
        const text = "A fantastic book! The author masterfully weaves a gripping narrative. Highly recommended!";
        const words = text.split(" ");
        for (const word of words) {
          await new Promise(r => setTimeout(r, 100));
          yield word + " ";
        }
      })();
    }
    return api.askLLMStream(book.title, book.author, "Why should I read this? Give a short 2-sentence pitch.");
  };

  const ctr = interactions.length
    ? (interactions.filter(i => i.action === "click" || i.action === "cart").length / interactions.length * 100).toFixed(1)
    : "—";

  // Pre-compute sparkline points once per loss_history change instead of on every render
  const sparklinePoints = useMemo(() => {
    const h = rlMetrics.loss_history;
    if (h.length < 2) return null;
    const min = Math.min(...h), max = Math.max(...h), range = max - min || 1;
    return h.map((v, i) => `${i},${1 - (v - min) / range}`).join(" ");
  }, [rlMetrics.loss_history]);

  // ── Shared class shorthands ──────────────────────────────────────────────────
  const CARD    = "rounded-xl border border-[#babbbd] dark:border-[#627d9a]/70 bg-white/50 dark:bg-[#fffef7]/5 shadow-sm";
  const DIVIDER = "border-[#babbbd] dark:border-[#627d9a]/60";

  // ─── RENDER ─────────────────────────────────────────────────────────────────

  // Gate: show login screen until a userId is established
  if (!userId) {
    return <LoginPage onLogin={handleLogin} isDark={isDark} onToggleDark={() => setIsDark(d => !d)} />;
  }

  return (
    <div className="h-screen flex flex-col font-sans bg-[#fffef7] dark:bg-[#2e3257] text-[#2e3257] dark:text-[#fffef7] overflow-hidden transition-colors duration-300">

      {/* Toast rack */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2" style={{ maxWidth: 300 }}>
        {toasts.map(t => (
          <div key={t.id} className={`fade-in px-3 py-2 rounded-xl text-[11px] border shadow-sm
            ${t.type === "success" ? "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-300 dark:border-emerald-700/60 text-emerald-700 dark:text-emerald-300"
            : t.type === "error"   ? "bg-red-50 dark:bg-red-900/30 border-red-300 dark:border-red-700/60 text-red-600 dark:text-red-300"
            : "bg-[#dfc5a4]/20 dark:bg-[#dfc5a4]/10 border-[#dfc5a4] text-[#627d9a] dark:text-[#babbbd]"}`}>
            {t.msg}
          </div>
        ))}
      </div>

      {/* ── HEADER ──────────────────────────────────────────────────────────── */}
      <header className={`relative z-10 flex-shrink-0 flex flex-col border-b ${DIVIDER} bg-[#fffef7]/90 dark:bg-[#2e3257]/90 backdrop-blur-md`}>
        <div className="flex items-center justify-between px-6 py-3">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 bg-[#2e3257] dark:bg-[#dfc5a4] shadow-sm select-none">
              <span className="font-serif font-bold text-[#fffef7] dark:text-[#2e3257]" style={{ fontSize: 20, lineHeight: 1 }}>愛</span>
            </div>
            <div>
              <h1 className="font-serif font-bold tracking-tight leading-none text-[#2e3257] dark:text-[#fffef7]" style={{ fontSize: 18 }}>
                Yomi<span className="text-[#627d9a] dark:text-[#dfc5a4]">愛</span>
              </h1>
              <p className="font-mono text-[#babbbd] dark:text-[#627d9a] mt-0.5 tracking-widest uppercase" style={{ fontSize: 8 }}>
                読み愛 · Multimodal Rec Engine
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-4">

            {/* Logged-in user badge + logout */}
            <div className="flex items-center gap-2">
              <span className={`px-3 py-1.5 rounded-lg text-[12px] font-semibold flex items-center gap-1.5 border
                ${isGuest
                  ? "bg-[#627d9a]/10 border-[#627d9a]/30 text-[#627d9a] dark:text-[#babbbd]"
                  : "bg-[#2e3257]/8 dark:bg-[#fffef7]/8 border-[#2e3257]/20 dark:border-[#fffef7]/20 text-[#2e3257] dark:text-[#fffef7]"}`}
              >
                <span style={{ fontSize: 14 }}>{isGuest ? "👤" : "🔖"}</span>
                {userId}
              </span>
              <button
                onClick={handleLogout}
                className="px-2 py-1.5 rounded-lg text-[11px] border border-[#babbbd]/50 dark:border-[#627d9a]/40
                           text-[#babbbd] dark:text-[#627d9a]
                           hover:border-rose-300 hover:text-rose-400 transition-all duration-200"
                title="Sign out"
              >
                ⎋
              </button>
            </div>

            <div className={`w-px h-6 ${DIVIDER} border-l`} />

            {/* Theme + Cart */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsDark(d => !d)}
                className={`px-3 py-1.5 rounded-lg text-[12px] border transition-all duration-200
                  hover:bg-[#dfc5a4]/30 hover:border-[#dfc5a4] hover:text-[#2e3257]
                  bg-transparent border-[#babbbd] dark:border-[#627d9a]/70
                  text-[#627d9a] dark:text-[#babbbd]`}
              >
                {isDark ? "☀ Light" : "◐ Dark"}
              </button>
              <button className="px-3 py-1.5 rounded-lg text-[12px] font-semibold flex items-center gap-1.5 border
                                 bg-transparent border-[#babbbd] dark:border-[#627d9a]/70
                                 text-[#2e3257] dark:text-[#fffef7]
                                 hover:bg-[#dfc5a4]/20 hover:border-[#dfc5a4] transition-all duration-200">
                🛒
                <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-[#2e3257] dark:bg-[#fffef7] text-[#fffef7] dark:text-[#2e3257]">
                  {cart.length}
                </span>
              </button>
            </div>

            <div className={`w-px h-6 ${DIVIDER} border-l`} />

            {/* Mock / Live */}
            <div className="flex items-center gap-2">
              <span className={`font-mono tracking-widest uppercase text-[9px] ${useMock ? "text-[#627d9a] dark:text-[#dfc5a4]" : "text-[#babbbd] dark:text-[#627d9a]/60"}`}>Mock</span>
              <button
                onClick={() => { setUseMock(m => !m); toast(useMock ? "Switched to Live API" : "Switched to Mock mode", "info"); }}
                className="relative rounded-full flex-shrink-0 transition-all duration-300"
                style={{
                  width: 36, height: 18,
                  background: useMock ? "#627d9a" : "#10b981",
                  border: `1px solid ${useMock ? "#babbbd" : "#6ee7b7"}`,
                }}
              >
                <div className="absolute rounded-full bg-white transition-all duration-300"
                  style={{ width: 12, height: 12, top: 2, left: useMock ? 2 : 20, boxShadow: "0 1px 3px rgba(0,0,0,0.3)" }} />
              </button>
              <span className={`font-mono tracking-widest uppercase text-[9px] ${!useMock ? "text-emerald-600 dark:text-emerald-400" : "text-[#babbbd] dark:text-[#627d9a]/60"}`}>Live</span>
            </div>

            <div className={`w-px h-6 ${DIVIDER} border-l`} />

            {/* Stats */}
            <div className="flex gap-4">
              {[
                { label: "Train Steps",   value: rlStep,             hi: rlStep > 0 },
                { label: "CTR",          value: `${ctr}%`,          hi: ctr !== "—" },
                { label: "Interactions", value: interactions.length, hi: interactions.length > 0 },
              ].map(s => (
                <div key={s.label} className="text-center">
                  <div className={`font-mono font-semibold tabular-nums ${s.hi ? "text-[#2e3257] dark:text-[#fffef7]" : "text-[#babbbd] dark:text-[#627d9a]"}`} style={{ fontSize: 15 }}>
                    {s.value}
                  </div>
                  <div className="font-mono tracking-widest uppercase text-[#babbbd] dark:text-[#627d9a]" style={{ fontSize: 8 }}>
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recently Viewed strip — live: server profile, mock: local interactions */}
        {(() => {
          const items = useMock
            ? interactions.slice(0, 8)
            : profileStats.recent_items;
          if (!items?.length) return null;
          return (
            <div className={`flex items-center gap-4 px-6 py-2 overflow-x-auto border-t ${DIVIDER} bg-[#babbbd]/10 dark:bg-[#fffef7]/3`}>
              <span className="font-mono tracking-widest uppercase text-[#babbbd] dark:text-[#627d9a] whitespace-nowrap font-bold flex-shrink-0" style={{ fontSize: 9 }}>
                Recently Viewed
              </span>
              {items.map((item, i) => (
                <div key={i} className="flex items-center gap-2 group cursor-pointer flex-shrink-0" title={item.title}>
                  {item.image_url ? (
                    <div className={`w-8 h-10 rounded overflow-hidden relative border ${DIVIDER} group-hover:border-[#dfc5a4] transition-colors`}>
                      <img src={item.image_url} alt="" className="w-full h-full object-cover" />
                    </div>
                  ) : (
                    <div className="w-8 h-10 rounded flex-shrink-0" style={{ background: item.cover_color || "#2e3257" }} />
                  )}
                  <div style={{ maxWidth: 80 }}>
                    <p className="truncate text-[#2e3257] dark:text-[#fffef7] font-medium" style={{ fontSize: 10 }}>{item.title}</p>
                    <p className="truncate text-[#babbbd] dark:text-[#627d9a]" style={{ fontSize: 8 }}>{item.author}</p>
                  </div>
                </div>
              ))}
            </div>
          );
        })()}
      </header>

      {/* ── MAIN ────────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">

        {/* ── LEFT PANEL (42%) ── */}
        <div className={`flex flex-col flex-shrink-0 border-r ${DIVIDER}`} style={{ width: "42%" }}>

          {/* Tab bar */}
          <div className={`flex px-4 pt-3 gap-1 border-b ${DIVIDER}`}>
            {[["search", "Active Search", "Mode 1"], ["recs", "Recommendations", "Mode 2"]].map(([tab, label, mode]) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2.5 text-center text-[11px] transition-all duration-200 rounded-t relative border-none cursor-pointer
                  ${activeTab === tab
                    ? "font-medium text-[#2e3257] dark:text-[#fffef7] bg-[#dfc5a4]/15 dark:bg-[#dfc5a4]/8"
                    : "font-normal text-[#627d9a] dark:text-[#babbbd] hover:text-[#2e3257] dark:hover:text-[#fffef7] bg-transparent"}`}
              >
                {label}
                <span className="text-[#babbbd] dark:text-[#627d9a] ml-1" style={{ fontSize: 9 }}>{mode}</span>
                {activeTab === tab && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t bg-[#2e3257] dark:bg-[#dfc5a4]" />
                )}
              </button>
            ))}
          </div>

          {/* ── Search tab ── */}
          {activeTab === "search" ? (
            <div className="flex flex-col flex-1 overflow-hidden px-4 pt-4 gap-3">

              {/* Input row */}
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && handleSearch()}
                    placeholder='"Dark Fantasy with complex magic systems"'
                    className={`flex-1 px-3 py-2.5 rounded-xl text-[12px] transition-all duration-200
                                bg-white dark:bg-[#fffef7]/5
                                border ${DIVIDER}
                                text-[#2e3257] dark:text-[#fffef7]
                                placeholder:text-[#babbbd] dark:placeholder:text-[#627d9a]
                                focus:border-[#2e3257] dark:focus:border-[#dfc5a4] focus:outline-none`}
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isSearching}
                    className={`px-5 py-2.5 rounded-xl text-[12px] font-semibold flex-shrink-0 transition-all duration-200
                                border border-transparent
                                ${isSearching
                                  ? "bg-[#babbbd]/30 text-[#627d9a] cursor-not-allowed"
                                  : "bg-[#2e3257] dark:bg-[#fffef7] text-[#fffef7] dark:text-[#2e3257] hover:bg-[#dfc5a4] hover:text-[#2e3257] shadow-sm"}`}
                  >
                    {isSearching ? <span className="shimmer">…</span> : "Search"}
                  </button>
                </div>

                {/* Image drop zone */}
                <div
                  ref={dropRef}
                  onClick={() => fileRef.current?.click()}
                  onDrop={handleImageDrop}
                  onDragOver={e => e.preventDefault()}
                  className={`flex items-center gap-3 rounded-xl cursor-pointer transition-all duration-200
                              border ${imagePreview ? "border-[#dfc5a4] bg-[#dfc5a4]/10" : `border-dashed border-[#babbbd] dark:border-[#627d9a]/60 bg-transparent hover:border-[#dfc5a4] hover:bg-[#dfc5a4]/5`}`}
                  style={{ padding: imagePreview ? "8px 12px" : "10px 14px" }}
                >
                  {imagePreview ? (
                    <>
                      <img src={imagePreview} className="w-11 h-11 rounded-lg object-cover flex-shrink-0 shadow-sm" alt="query" />
                      <div className="flex-1 min-w-0">
                        <p className="text-[11px] font-medium text-[#2e3257] dark:text-[#fffef7]">Image query loaded</p>
                        <p className="text-[9px] text-[#627d9a] dark:text-[#babbbd]">CLIP will encode this for visual similarity</p>
                      </div>
                      <button onClick={e => { e.stopPropagation(); setImageFile(null); setImagePreview(null); }}
                        className="text-[#babbbd] hover:text-[#627d9a] dark:hover:text-[#babbbd] transition-colors flex-shrink-0 bg-transparent border-none cursor-pointer p-1" style={{ fontSize: 14 }}>
                        ✕
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="text-[#babbbd] dark:text-[#627d9a] flex-shrink-0" style={{ fontSize: 20, opacity: 0.6 }}>🖼</span>
                      <div>
                        <p className="text-[11px] text-[#627d9a] dark:text-[#babbbd]">Drop a cover image here</p>
                        <p className="text-[9px] text-[#babbbd] dark:text-[#627d9a]/70">CLIP image encoder · visual similarity search</p>
                      </div>
                    </>
                  )}
                  <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleImageDrop} />
                </div>
              </div>

              {/* Encoder legend */}
              <div className="flex items-center gap-4">
                {[["BGE-M3", "#2e3257", "text semantics"], ["CLIP", "#627d9a", "visual features"], ["RRF", "#dfc5a4", "fusion score"]].map(([name, color, desc]) => (
                  <div key={name} className="flex items-center gap-1.5">
                    <div className="rounded-full flex-shrink-0" style={{ width: 6, height: 6, background: color }} />
                    <span className="text-[#627d9a] dark:text-[#babbbd]" style={{ fontSize: 9 }}>
                      <span className="font-medium text-[#2e3257] dark:text-[#fffef7]">{name}</span> · {desc}
                    </span>
                  </div>
                ))}
              </div>

              {/* Results */}
              <div className="flex-1 overflow-y-auto space-y-2 pr-1 pb-2">
                {isSearching ? (
                  [0,1,2,4].map(i => <SkeletonCard key={i} size="md" />)
                ) : searchError ? (
                  <div className={`p-4 rounded-xl border ${DIVIDER} text-center space-y-2`}>
                    <p className="text-[12px] font-medium text-red-500 dark:text-red-400">{searchError}</p>
                    <button
                      onClick={handleSearch}
                      className="text-[11px] px-3 py-1.5 rounded-lg border border-[#babbbd] dark:border-[#627d9a]/70
                                 text-[#627d9a] dark:text-[#babbbd] hover:bg-[#dfc5a4]/20 hover:border-[#dfc5a4] transition-all"
                    >
                      ↺ Retry
                    </button>
                  </div>
                ) : searchResults.length > 0 ? (
                  searchResults.map((book, i) => (
                    <SearchResultCard key={i} book={book} onInteract={handleInteract} onAskAIStream={handleAskAIStream} />
                  ))
                ) : (
                  <div className="flex flex-col items-center justify-center h-48 gap-3 text-center">
                    <div className="text-[#babbbd] dark:text-[#627d9a]" style={{ fontSize: 40, opacity: 0.4 }}>📖</div>
                    <div>
                      <p className="text-[13px] font-medium text-[#627d9a] dark:text-[#babbbd]">Search for a book to begin</p>
                      <p className="text-[10px] text-[#babbbd] dark:text-[#627d9a]/70 mt-1">BGE-M3 + CLIP · 3M book catalog</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

          ) : (
            /* ── Recs tab ── */
            <div className="flex flex-col flex-1 overflow-hidden px-4 pt-4 gap-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[12px] font-medium text-[#2e3257] dark:text-[#fffef7]">Personalized For You</p>
                  <p className="text-[10px] text-[#627d9a] dark:text-[#babbbd] mt-0.5">Retrieval + DIF-SASRec Multi-mode</p>
                </div>
                <button
                  onClick={loadRecs}
                  disabled={isLoadingRecs}
                  className={`text-[11px] px-3 py-1.5 rounded-lg border transition-all duration-200
                    ${isLoadingRecs
                      ? "border-[#babbbd] dark:border-[#627d9a]/50 text-[#babbbd] cursor-not-allowed"
                      : `border-[#babbbd] dark:border-[#627d9a]/70 text-[#627d9a] dark:text-[#babbbd]
                         hover:bg-[#dfc5a4]/20 hover:border-[#dfc5a4] hover:text-[#2e3257] dark:hover:text-[#fffef7]`}`}
                >
                  {isLoadingRecs ? (
                    <span className="shimmer">Fetching…</span>
                  ) : (
                    <span>
                      ↺ Refresh
                      {lastRecsRefresh && (
                        <span className="ml-1.5 text-[#babbbd] dark:text-[#627d9a] font-normal" style={{ fontSize: 9 }}>
                          {lastRecsRefresh.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      )}
                    </span>
                  )}
                </button>
              </div>

              {/* Sub-tabs */}
              <div className={`flex rounded-xl p-1 border ${DIVIDER} bg-[#babbbd]/10 dark:bg-[#fffef7]/5`}>
                {[
                  { id: "pab", label: "People Also Buy", desc: "Cleora + BGE-M3/CLIP" },
                  { id: "yml", label: "You Might Like",  desc: "DIF-SASRec Personalized" },
                ].map(t => (
                  <button
                    key={t.id}
                    onClick={() => setRecTab(t.id)}
                    className={`flex-1 flex flex-col items-center py-1.5 rounded-lg transition-all duration-200 border
                      ${recTab === t.id
                        ? "bg-[#fffef7] dark:bg-[#2e3257]/80 border-[#babbbd] dark:border-[#627d9a]/60 shadow-sm"
                        : "bg-transparent border-transparent hover:bg-[#dfc5a4]/10"}`}
                  >
                    <span className={`text-[11px] ${recTab === t.id ? "font-semibold text-[#2e3257] dark:text-[#fffef7]" : "font-normal text-[#627d9a] dark:text-[#babbbd]"}`}>{t.label}</span>
                    <span className="text-[9px] text-[#babbbd] dark:text-[#627d9a]">{t.desc}</span>
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                {isLoadingRecs ? (
                  [0,1,2,3].map(i => <SkeletonCard key={i} size="sm" />)
                ) : recsError ? (
                  <div className={`p-4 rounded-xl border ${DIVIDER} text-center space-y-2`}>
                    <p className="text-[12px] font-medium text-red-500 dark:text-red-400">{recsError}</p>
                    <button
                      onClick={loadRecs}
                      className="text-[11px] px-3 py-1.5 rounded-lg border border-[#babbbd] dark:border-[#627d9a]/70
                                 text-[#627d9a] dark:text-[#babbbd] hover:bg-[#dfc5a4]/20 hover:border-[#dfc5a4] transition-all"
                    >
                      ↺ Retry
                    </button>
                  </div>
                ) : (
                  (recTab === "pab" ? recommendations.people_also_buy : recommendations.you_might_like)?.map((book, i) => (
                    <RecommendCard key={i} book={book} rank={i} onInteract={handleInteract} onAskAIStream={handleAskAIStream} />
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div className="flex flex-col flex-1 overflow-hidden min-w-0">

          {/* Tabs */}
          <div className={`flex px-4 pt-3 gap-1 border-b ${DIVIDER}`}>
            {[["profile", "User Profile"], ["history", "Activity History"]].map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setActiveRight(tab)}
                className={`flex-1 py-2.5 text-center text-[11px] transition-all duration-200 rounded-t relative border-none cursor-pointer
                  ${activeRight === tab
                    ? "font-medium text-[#2e3257] dark:text-[#fffef7] bg-[#dfc5a4]/15 dark:bg-[#dfc5a4]/8"
                    : "font-normal text-[#627d9a] dark:text-[#babbbd] hover:text-[#2e3257] dark:hover:text-[#fffef7] bg-transparent"}`}
              >
                {label}
                {activeRight === tab && <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t bg-[#2e3257] dark:bg-[#dfc5a4]" />}
              </button>
            ))}
          </div>

          {/* ── Profile tab — bento grid ── */}
          {activeRight === "profile" ? (
            <div className="flex-1 overflow-y-auto p-4">
              {/* User identity row */}
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-[13px] font-extrabold tracking-tight text-[#2e3257] dark:text-[#fffef7]">User Profile State</h2>
                  <p className="text-[10px] text-[#627d9a] dark:text-[#babbbd] mt-0.5">
                    Aggregated embedding · temporal decay λ=0.1
                  </p>
                </div>
                <div className="text-right">
                  <div className="font-mono text-[10px] text-[#627d9a] dark:text-[#babbbd]">{userId}</div>
                  {lastTrained && <div className="font-mono text-[9px] text-[#babbbd] dark:text-[#627d9a] mt-0.5">updated {lastTrained.toLocaleTimeString()}</div>}
                </div>
              </div>

              {/* ── Bento grid ── */}
              <div className="grid grid-cols-2 gap-3">

                {/* Radar — full width */}
                <div className={`col-span-2 p-4 ${CARD}`}>
                  <p className="text-[10px] font-semibold tracking-widest uppercase text-[#babbbd] dark:text-[#627d9a] mb-3">Engagement Signals</p>
                  <ProfileRadar interactions={interactions} />
                </div>

                {/* SASRec Training Feed */}
                <div className={`p-3 ${CARD}`}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-[10px] font-semibold tracking-widest uppercase text-[#babbbd] dark:text-[#627d9a]">Train Feed</p>
                    <div className="flex items-center gap-1.5">
                      <div className={`w-1.5 h-1.5 rounded-full ${rlStep > 0 ? "bg-emerald-500 live-dot" : "bg-[#babbbd] dark:bg-[#627d9a]"}`} />
                      <span className="font-mono text-[#babbbd] dark:text-[#627d9a]" style={{ fontSize: 9 }}>{rlStep} steps</span>
                    </div>
                  </div>
                  <div className="space-y-1.5 overflow-y-auto" style={{ maxHeight: 110 }}>
                    {interactions.slice(0, 6).map((item, i) => (
                      <div key={i} className="flex items-center gap-2 fade-in">
                        <span className={`font-mono text-[9px] min-w-[60px] ${(item.action === "click" || item.action === "cart") ? "text-emerald-600 dark:text-emerald-400" : "text-red-500 dark:text-red-400"}`}>
                          {(item.action === "click" || item.action === "cart") ? "▲ trained" : "▼ skipped"}
                        </span>
                        <span className="flex-1 truncate font-mono text-[9px] text-[#babbbd] dark:text-[#627d9a]">{item.title}</span>
                      </div>
                    ))}
                    {interactions.length === 0 && (
                      <p className="text-[10px] text-[#babbbd] dark:text-[#627d9a]">No interactions yet — click or skip items to train</p>
                    )}
                  </div>
                </div>

                {/* SASRec Loss */}
                <div className={`p-3 ${CARD}`}>
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <p className="text-[10px] font-semibold tracking-widest uppercase text-[#babbbd] dark:text-[#627d9a]">SASRec Loss</p>
                      <p className="font-mono text-[9px] text-[#babbbd] dark:text-[#627d9a] mt-0.5">{rlMetrics.step} online steps</p>
                    </div>
                    <span className="font-mono font-semibold text-[#2e3257] dark:text-[#dfc5a4]" style={{ fontSize: 13 }}>
                      {rlMetrics.loss_history.length > 0 ? rlMetrics.loss_history.at(-1).toFixed(4) : "0.0000"}
                    </span>
                  </div>
                  {sparklinePoints ? (
                    <div className="h-14 w-full relative" style={{ borderBottom: "1px solid #babbbd55" }}>
                      <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none"
                        viewBox={`0 0 ${Math.max(1, rlMetrics.loss_history.length - 1)} 1`}>
                        <polyline
                          fill="none" stroke="#627d9a" strokeWidth="0.05" vectorEffect="non-scaling-stroke"
                          points={sparklinePoints}
                        />
                      </svg>
                    </div>
                  ) : (
                    <div className={`h-14 flex items-center justify-center rounded-lg border ${DIVIDER}`}>
                      <span className="text-[9px] text-[#babbbd] dark:text-[#627d9a]">Interact to see loss converge</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

          ) : (
            /* ── History tab ── */
            <div className="flex-1 overflow-y-auto px-5 py-4">
              <h3 className="text-[11px] font-extrabold tracking-widest uppercase text-[#babbbd] dark:text-[#627d9a] mb-3">
                Interaction History
              </h3>
              {interactions.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-40 gap-3 text-center">
                  <p className="text-[11px] text-[#babbbd] dark:text-[#627d9a]">
                    Interact with search results or<br />recommendations to train DIF-SASRec
                  </p>
                  <p className="text-[9px] text-[#babbbd]/70 dark:text-[#627d9a]/60 mt-1">Profile fingerprint updates in real-time</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {interactions.map((item, i) => (
                    <div key={i} className={`flex items-center gap-3 py-2 fade-in border-b ${DIVIDER}`}>
                      <BookCover color={item.cover_color} title={item.title} size="sm" imageUrl={item.image_url} />
                      <div className="flex-1 min-w-0">
                        <p className="truncate font-serif text-[11px] font-medium text-[#2e3257] dark:text-[#fffef7]">{item.title}</p>
                        <p className="text-[9px] text-[#627d9a] dark:text-[#babbbd] mt-0.5">{item.author}</p>
                      </div>
                      <div className="flex flex-col items-end gap-1 flex-shrink-0">
                        <span className={`text-[9px] px-2 py-0.5 rounded-full font-medium border
                          ${(item.action === "click" || item.action === "cart")
                            ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700/60 text-emerald-700 dark:text-emerald-400"
                            : "bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700/60 text-red-600 dark:text-red-400"}`}>
                          {item.action}
                        </span>
                        <span className="font-mono text-[8px] text-[#babbbd] dark:text-[#627d9a]">
                          {new Date(item.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
