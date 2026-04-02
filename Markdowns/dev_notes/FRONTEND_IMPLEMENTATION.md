# NBA System — Frontend & API Integration Spec
> **For the AI agent**: This document is a complete, self-contained implementation guide. Follow each section in order. All file paths are relative to the project root. All stub replacement points are marked with `TODO:` comments referencing exact existing source files.

---

## 0. Context & Goals

This project is a **Dual-Mode Multimodal Book Recommendation System** (DATN capstone). The core ML pipeline already exists. The goal of this task is to:

1. Scaffold a **React + Vite frontend** (`frontend/`) that lets a user run the full simulation interactively.
2. Add a **FastAPI backend** (`api.py`) at the project root that exposes the existing pipeline over HTTP.
3. Wire the two together so every user interaction (click/skip) updates the user profile and triggers a DQN `train_step()` in real time.

The existing pipeline lives in `src/`. **Do not modify any existing source files** — only add new files.

---

## 1. Project Structure After Implementation

```
<project_root>/
├── api.py                        ← NEW: FastAPI server (4 endpoints)
├── frontend/                     ← NEW: Vite + React app
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       └── App.jsx               ← NEW: Full UI (provided below)
├── src/                          ← EXISTING: do not modify
│   ├── data/
│   │   └── books_meta.zip        ← Arrow files: book metadata
│   ├── retrieval/                ← BLaIR + CLIP FAISS search logic
│   ├── behavioral/               ← Cleora embeddings + neighbor search
│   ├── rl/                       ← DQN agent + train_step
│   └── profile/                  ← UserProfileManager (temporal decay)
└── main.py                       ← EXISTING: simulation entrypoint
```

---

## 2. Step 1 — Scaffold the Frontend

Run these commands from the **project root**:

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

Then replace `frontend/src/App.jsx` with the full component in **Section 5** of this document.

Replace `frontend/src/main.jsx` with:

```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

Replace `frontend/index.html` `<title>` with `NBA System — DATN Demo`.

Add a proxy to `frontend/vite.config.js` so API calls work without CORS issues in dev:

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

> **Note**: The `App.jsx` uses `API_BASE = ""` when the proxy is active. Update the constant at the top of `App.jsx` if you want to point directly at `http://localhost:8000` instead.

---

## 3. Step 2 — Create `api.py` at Project Root

Create the file `api.py` with the content in **Section 6**. Then install dependencies:

```bash
pip install fastapi uvicorn pillow numpy
```

Run the server with:

```bash
uvicorn api:app --reload --port 8000
```

---

## 4. Step 3 — Wire the Stubs

`api.py` has **4 stub blocks** that need to be replaced with calls to the existing `src/` modules. Each is marked `# TODO:` with the exact function signatures to call. Here is a summary:

| Endpoint | Stub location | What to replace with |
|---|---|---|
| `POST /search` | `api.py` line ~70 | `MultimodalSearchPipeline.search_text()` + `search_image()` + `rrf_fusion()` from `src/retrieval/` |
| `GET /recommend` | `api.py` line ~100 | `cleora_neighbors()` → cosine veto loop → `DQNAgent.q_value()` from `src/behavioral/` + `src/rl/` |
| `POST /interact` | `api.py` line ~130 | `UserProfileManager.update()` + `DQNAgent.train_step()` from `src/profile/` + `src/rl/` |
| `GET /profile` | `api.py` line ~155 | `UserProfileManager.get()` stats from `src/profile/` |

> **Agent instruction**: Before replacing stubs, read the relevant `src/` files to confirm exact class names, method signatures, and constructor arguments. Do not assume — inspect first.

---

## 5. `frontend/src/App.jsx` — Full Source

```jsx
import { useState, useEffect, useRef, useCallback } from "react";

// ─── API CONFIG ───────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

async function apiSearch(query, imageBase64 = null) {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, image_base64: imageBase64, top_k: 10 }),
  });
  return res.json();
}

async function apiRecommend(userId) {
  const res = await fetch(`${API_BASE}/recommend?user_id=${userId}`);
  return res.json();
}

async function apiInteract(userId, itemId, action) {
  const res = await fetch(`${API_BASE}/interact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, item_id: itemId, action }),
  });
  return res.json();
}

async function apiProfile(userId) {
  const res = await fetch(`${API_BASE}/profile?user_id=${userId}`);
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

const MOCK_RECS = [
  { id: "r001", title: "The Lies of Locke Lamora", author: "Scott Lynch", genre: "Fantasy Heist", score: 0.91, cover_color: "#0a1a2d", layer: "Cleora + BLaIR" },
  { id: "r002", title: "Assassin's Apprentice", author: "Robin Hobb", genre: "Epic Fantasy", score: 0.88, cover_color: "#0d2010", layer: "Cleora + CLIP" },
  { id: "r003", title: "The Black Prism", author: "Brent Weeks", genre: "Dark Fantasy", score: 0.86, cover_color: "#1a1000", layer: "RL-DQN" },
];

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function BookCover({ color, title, size = "md" }) {
  return (
    <div
      className="rounded-sm flex-shrink-0 relative overflow-hidden"
      style={{
        background: `linear-gradient(135deg, ${color}ee, ${color}88)`,
        boxShadow: `2px 3px 12px ${color}66, inset -2px 0 6px rgba(0,0,0,0.4)`,
        width: size === "lg" ? 80 : size === "md" ? 64 : 40,
        height: size === "lg" ? 112 : size === "md" ? 88 : 56,
      }}
    >
      <div className="absolute inset-0" style={{ background: "linear-gradient(to right, rgba(255,255,255,0.08) 0%, transparent 30%)" }} />
      <div className="absolute bottom-0 left-0 right-0 p-1">
        <div className="text-white/60 font-mono" style={{ fontSize: 5, lineHeight: 1.3 }}>{title?.slice(0, 20)}</div>
      </div>
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
          <circle cx="18" cy="18" r="14" fill="none"
            stroke={pct > 85 ? "#a78bfa" : pct > 70 ? "#60a5fa" : "#94a3b8"}
            strokeWidth="3" strokeDasharray={`${pct * 0.88} 88`} strokeLinecap="round" />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-white font-mono" style={{ fontSize: 9 }}>{pct}</span>
      </div>
      <span className="text-white/40 mt-0.5" style={{ fontSize: 9 }}>{label}</span>
    </div>
  );
}

function LayerTag({ label }) {
  const colors = { "Cleora + BLaIR": "rgba(167,139,250,0.15)", "Cleora + CLIP": "rgba(96,165,250,0.15)", "RL-DQN": "rgba(52,211,153,0.15)" };
  const border = { "Cleora + BLaIR": "rgba(167,139,250,0.4)", "Cleora + CLIP": "rgba(96,165,250,0.4)", "RL-DQN": "rgba(52,211,153,0.4)" };
  return (
    <span className="text-white/70 px-2 py-0.5 rounded-full"
      style={{ fontSize: 9, background: colors[label] || "rgba(255,255,255,0.05)", border: `1px solid ${border[label] || "rgba(255,255,255,0.1)"}` }}>
      {label}
    </span>
  );
}

function ProfileRadar({ interactions }) {
  const total = interactions.length;
  const clicks = interactions.filter(i => i.action === "click").length;
  const bars = [
    { label: "CTR", value: total > 0 ? clicks / total : 0 },
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
            <div className="h-1.5 rounded-full transition-all duration-700"
              style={{ width: `${b.value * 100}%`, background: "linear-gradient(to right, #7c3aed, #a78bfa)" }} />
          </div>
          <span className="text-white/30 font-mono w-8" style={{ fontSize: 10 }}>{Math.round(b.value * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

function SearchResultCard({ book, onInteract }) {
  return (
    <div className="flex gap-3 p-3 rounded-lg cursor-pointer transition-all duration-300"
      style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
      onMouseEnter={e => e.currentTarget.style.background = "rgba(167,139,250,0.08)"}
      onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.04)"}>
      <BookCover color={book.cover_color} title={book.title} size="md" />
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="text-white/90 font-medium leading-tight" style={{ fontFamily: "'Playfair Display', Georgia, serif", fontSize: 13 }}>{book.title}</p>
            <p className="text-white/40 mt-0.5" style={{ fontSize: 11 }}>{book.author}</p>
            <span className="inline-block mt-1 px-2 py-0.5 rounded-full text-white/50" style={{ fontSize: 9, background: "rgba(255,255,255,0.06)" }}>{book.genre}</span>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <ScoreBadge score={book.text_sim || book.score} label="Text" />
            <ScoreBadge score={book.img_sim || book.score * 0.95} label="Img" />
          </div>
        </div>
        <div className="flex gap-2 mt-2">
          <button onClick={() => onInteract(book, "click")} className="flex-1 py-1.5 rounded text-white/80 font-medium transition-all duration-200 hover:text-white"
            style={{ fontSize: 11, background: "linear-gradient(135deg, rgba(124,58,237,0.3), rgba(167,139,250,0.2))", border: "1px solid rgba(167,139,250,0.3)" }}>
            👆 Click
          </button>
          <button onClick={() => onInteract(book, "skip")} className="flex-1 py-1.5 rounded text-white/40 transition-all duration-200 hover:text-white/60"
            style={{ fontSize: 11, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}>
            ✕ Skip
          </button>
        </div>
      </div>
    </div>
  );
}

function RecommendCard({ book, onInteract, rank }) {
  return (
    <div className="flex gap-3 p-3 rounded-lg cursor-pointer transition-all duration-300"
      style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
      onMouseEnter={e => { e.currentTarget.style.background = "rgba(52,211,153,0.06)"; e.currentTarget.style.borderColor = "rgba(52,211,153,0.2)"; }}
      onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.03)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; }}>
      <div className="flex flex-col items-center gap-2">
        <span className="text-white/20 font-mono font-bold" style={{ fontSize: 10 }}>#{rank + 1}</span>
        <BookCover color={book.cover_color} title={book.title} size="sm" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white/85 font-medium" style={{ fontFamily: "'Playfair Display', Georgia, serif", fontSize: 12 }}>{book.title}</p>
        <p className="text-white/35 mt-0.5" style={{ fontSize: 10 }}>{book.author}</p>
        <div className="flex items-center gap-2 mt-1.5">
          {book.layer && <LayerTag label={book.layer} />}
          <span className="text-white/30 font-mono" style={{ fontSize: 9 }}>score: {(book.score * 100).toFixed(0)}</span>
        </div>
        <div className="flex gap-1.5 mt-2">
          <button onClick={() => onInteract(book, "click")} className="flex-1 py-1 rounded text-white/70 transition-colors duration-150 hover:text-white"
            style={{ fontSize: 10, background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.25)" }}>
            ✓ Interested
          </button>
          <button onClick={() => onInteract(book, "skip")} className="flex-1 py-1 rounded text-white/30 transition-colors duration-150 hover:text-white/50"
            style={{ fontSize: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}>
            ✗ Not for me
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [userId] = useState("user_demo_01");
  const [query, setQuery] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [interactions, setInteractions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingRecs, setIsLoadingRecs] = useState(false);
  const [rlStep, setRlStep] = useState(0);
  const [lastTrained, setLastTrained] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [useMock, setUseMock] = useState(true);
  const [activeTab, setActiveTab] = useState("search");
  const fileInputRef = useRef();

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
        const data = await apiSearch(query, imgB64);
        setSearchResults(data.results || []);
        addToast(`Found ${data.results?.length} results`, "success");
      }
    } catch (e) {
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
        setRecommendations([...MOCK_RECS].sort(() => 0.5 - Math.random()));
        addToast("3-layer funnel complete: Cleora → Veto → DQN", "success");
      } else {
        const data = await apiRecommend(userId);
        setRecommendations(data.recommendations || []);
        addToast("Personalized recommendations loaded", "success");
      }
    } catch (e) {
      addToast("Backend unreachable — enable mock mode", "error");
    } finally {
      setIsLoadingRecs(false);
    }
  };

  const handleInteract = async (book, action) => {
    setInteractions(prev => [{ ...book, action, ts: Date.now() }, ...prev]);
    setRlStep(s => s + 1);
    addToast(action === "click" ? `📚 Clicked "${book.title}" — profile updated` : `✕ Skipped "${book.title}" — RL penalized`, action === "click" ? "success" : "info");
    try {
      if (!useMock) await apiInteract(userId, book.id, action);
      setLastTrained(new Date());
      if ((rlStep + 1) % 3 === 0) setTimeout(() => loadRecommendations(), 800);
    } catch (e) {}
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

  useEffect(() => { loadRecommendations(); }, []);

  const ctr = interactions.length > 0 ? (interactions.filter(i => i.action === "click").length / interactions.length * 100).toFixed(1) : "—";

  return (
    <div className="min-h-screen text-white" style={{ background: "#080b14", fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@400;600;700&family=DM+Mono:wght@300;400&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 2px; }
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes shimmer { 0%,100% { opacity: 0.4; } 50% { opacity: 0.8; } }
        .fade-in { animation: fadeSlideIn 0.3s ease forwards; }
        .shimmer { animation: shimmer 1.5s ease infinite; }
      `}</style>

      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2" style={{ maxWidth: 280 }}>
        {toasts.map(t => (
          <div key={t.id} className="fade-in px-3 py-2 rounded-lg" style={{
            background: t.type === "success" ? "rgba(52,211,153,0.15)" : t.type === "error" ? "rgba(239,68,68,0.15)" : "rgba(167,139,250,0.15)",
            border: `1px solid ${t.type === "success" ? "rgba(52,211,153,0.3)" : t.type === "error" ? "rgba(239,68,68,0.3)" : "rgba(167,139,250,0.3)"}`,
            color: t.type === "success" ? "#6ee7b7" : t.type === "error" ? "#fca5a5" : "#c4b5fd",
            fontSize: 11,
          }}>{t.msg}</div>
        ))}
      </div>

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg, #7c3aed, #4f46e5)" }}>
            <span style={{ fontSize: 14 }}>📖</span>
          </div>
          <div>
            <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 16, fontWeight: 600 }}>
              NBA<span style={{ color: "#a78bfa" }}>sys</span>
            </h1>
            <p className="text-white/30" style={{ fontSize: 9, marginTop: -2, fontFamily: "DM Mono, monospace" }}>DATN · Multimodal Recommendation Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-white/30" style={{ fontSize: 10, fontFamily: "DM Mono, monospace" }}>MOCK</span>
            <button onClick={() => { setUseMock(m => !m); addToast(useMock ? "Switched to Live API" : "Switched to Mock", "info"); }}
              className="relative w-8 h-4 rounded-full transition-colors duration-300"
              style={{ background: useMock ? "rgba(167,139,250,0.4)" : "rgba(52,211,153,0.4)", border: "none", cursor: "pointer" }}>
              <div className="absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all duration-300"
                style={{ left: useMock ? 2 : 18, boxShadow: "0 1px 3px rgba(0,0,0,0.4)" }} />
            </button>
            <span className="text-white/30" style={{ fontSize: 10, fontFamily: "DM Mono, monospace" }}>LIVE</span>
          </div>
          <div className="flex gap-4">
            {[{ label: "RL Steps", value: rlStep }, { label: "CTR", value: `${ctr}%` }, { label: "Interactions", value: interactions.length }].map(s => (
              <div key={s.label} className="text-center">
                <div className="text-white/80 font-mono font-medium" style={{ fontSize: 13 }}>{s.value}</div>
                <div className="text-white/25" style={{ fontSize: 9 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex" style={{ height: "calc(100vh - 65px)" }}>

        {/* LEFT — Mode 1 / Mode 2 tabs */}
        <div className="flex flex-col" style={{ width: "42%", borderRight: "1px solid rgba(255,255,255,0.06)" }}>
          <div className="flex px-4 pt-3">
            {[["search", "🔍 Active Search", "Mode 1"], ["recs", "✨ Recommendations", "Mode 2"]].map(([tab, label, mode]) => (
              <button key={tab} onClick={() => setActiveTab(tab)} className="flex-1 py-2 text-center transition-all duration-200"
                style={{ fontSize: 11, fontWeight: activeTab === tab ? 500 : 400, color: activeTab === tab ? "#c4b5fd" : "rgba(255,255,255,0.3)", background: "transparent", border: "none", borderBottom: `2px solid ${activeTab === tab ? "#7c3aed" : "transparent"}`, cursor: "pointer" }}>
                {label} <span className="text-white/20" style={{ fontSize: 9 }}>{mode}</span>
              </button>
            ))}
          </div>

          {activeTab === "search" ? (
            <div className="flex flex-col flex-1 overflow-hidden px-4 pt-3 gap-3">
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === "Enter" && handleSearch()}
                    placeholder='"Dark Fantasy with complex magic systems"'
                    className="flex-1 px-3 py-2.5 rounded-lg text-white/80 placeholder-white/20 outline-none"
                    style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", fontSize: 12 }}
                    onFocus={e => e.target.style.borderColor = "rgba(167,139,250,0.5)"}
                    onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.1)"} />
                  <button onClick={handleSearch} disabled={isSearching}
                    className="px-4 py-2.5 rounded-lg font-medium transition-all duration-200"
                    style={{ fontSize: 12, background: isSearching ? "rgba(167,139,250,0.2)" : "linear-gradient(135deg, #7c3aed, #6d28d9)", border: "1px solid rgba(167,139,250,0.3)", cursor: isSearching ? "not-allowed" : "pointer", color: "white" }}>
                    {isSearching ? <span className="shimmer">…</span> : "Search"}
                  </button>
                </div>
                <div onClick={() => fileInputRef.current?.click()} onDrop={handleImageDrop} onDragOver={e => e.preventDefault()}
                  className="relative rounded-lg flex items-center gap-3 cursor-pointer transition-all duration-200"
                  style={{ padding: imagePreview ? "8px" : "10px 12px", background: imagePreview ? "rgba(167,139,250,0.08)" : "rgba(255,255,255,0.02)", border: `1px dashed ${imagePreview ? "rgba(167,139,250,0.4)" : "rgba(255,255,255,0.1)"}` }}>
                  {imagePreview ? (
                    <>
                      <img src={imagePreview} className="w-12 h-12 rounded object-cover" alt="query" />
                      <div>
                        <p className="text-white/60" style={{ fontSize: 11 }}>Image query loaded</p>
                        <p className="text-white/30" style={{ fontSize: 9 }}>CLIP will encode this for visual similarity</p>
                      </div>
                      <button onClick={e => { e.stopPropagation(); setImageFile(null); setImagePreview(null); }}
                        className="ml-auto text-white/30 hover:text-white/60" style={{ fontSize: 14, background: "none", border: "none", cursor: "pointer" }}>✕</button>
                    </>
                  ) : (
                    <>
                      <span style={{ fontSize: 18, opacity: 0.4 }}>🖼</span>
                      <div>
                        <p className="text-white/35" style={{ fontSize: 11 }}>Drop a cover image here</p>
                        <p className="text-white/20" style={{ fontSize: 9 }}>CLIP image encoder · visual similarity search</p>
                      </div>
                    </>
                  )}
                  <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleImageDrop} />
                </div>
              </div>
              <div className="flex gap-3">
                {[["BLaIR", "#7c3aed", "text semantics"], ["CLIP", "#2563eb", "visual features"], ["RRF Fusion", "#059669", "final score"]].map(([name, color, desc]) => (
                  <div key={name} className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                    <span className="text-white/40" style={{ fontSize: 9 }}><span style={{ color: "rgba(255,255,255,0.6)" }}>{name}</span> · {desc}</span>
                  </div>
                ))}
              </div>
              <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                {isSearching ? (
                  <div className="flex flex-col items-center justify-center h-32 gap-2">
                    <div className="flex gap-1">{[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full" style={{ background: "#7c3aed", animation: `shimmer 1s ease ${i*0.2}s infinite` }} />)}</div>
                    <p className="text-white/30" style={{ fontSize: 11 }}>Encoding query → FAISS search…</p>
                  </div>
                ) : searchResults.length > 0 ? searchResults.map((book, i) => (
                  <div key={book.id} className="fade-in" style={{ animationDelay: `${i * 50}ms` }}>
                    <SearchResultCard book={book} onInteract={handleInteract} />
                  </div>
                )) : (
                  <div className="flex flex-col items-center justify-center h-40 text-center">
                    <div style={{ fontSize: 32, opacity: 0.2 }}>📚</div>
                    <p className="text-white/25 mt-2" style={{ fontSize: 12 }}>Search for a book to begin</p>
                    <p className="text-white/15 mt-1" style={{ fontSize: 10 }}>BLaIR + CLIP · 3M book catalog</p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-col flex-1 overflow-hidden px-4 pt-3 gap-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white/60" style={{ fontSize: 12 }}>3-Layer Funnel Output</p>
                  <p className="text-white/25" style={{ fontSize: 10 }}>Cleora → Content Veto → RL-DQN reranking</p>
                </div>
                <button onClick={loadRecommendations} disabled={isLoadingRecs}
                  className="px-3 py-1.5 rounded-lg text-white/60 transition-all duration-200 hover:text-white"
                  style={{ fontSize: 11, background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.25)", cursor: "pointer" }}>
                  {isLoadingRecs ? <span className="shimmer">Fetching…</span> : "↺ Refresh"}
                </button>
              </div>
              <div className="flex gap-1.5">
                {[["L1", "Behavioral", "#7c3aed"], ["L2", "Veto", "#2563eb"], ["L3", "DQN", "#059669"]].map(([l, name, color]) => (
                  <div key={l} className="flex-1 px-2 py-1.5 rounded text-center" style={{ background: `${color}18`, border: `1px solid ${color}33` }}>
                    <div className="font-mono font-bold" style={{ fontSize: 9, color }}>{l}</div>
                    <div className="text-white/40" style={{ fontSize: 9 }}>{name}</div>
                  </div>
                ))}
              </div>
              <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                {isLoadingRecs ? (
                  <div className="flex flex-col items-center justify-center h-32 gap-2">
                    <div className="flex gap-1">{[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full" style={{ background: "#34d399", animation: `shimmer 1s ease ${i*0.2}s infinite` }} />)}</div>
                    <p className="text-white/30" style={{ fontSize: 11 }}>Running 3-layer funnel…</p>
                  </div>
                ) : recommendations.map((book, i) => (
                  <div key={book.id} className="fade-in" style={{ animationDelay: `${i * 60}ms` }}>
                    <RecommendCard book={book} onInteract={handleInteract} rank={i} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT — Profile + Activity */}
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="px-5 py-4" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-white/80 font-medium" style={{ fontSize: 13 }}>User Profile State</h2>
                <p className="text-white/25" style={{ fontSize: 10 }}>Aggregated embedding · temporal decay λ=0.1</p>
              </div>
              <div className="text-right">
                <div className="text-white/60 font-mono" style={{ fontSize: 10 }}>{userId}</div>
                {lastTrained && <div className="text-white/25" style={{ fontSize: 9 }}>last RL update: {lastTrained.toLocaleTimeString()}</div>}
              </div>
            </div>
            <ProfileRadar interactions={interactions} />
          </div>
          <div className="px-5 py-3" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-white/50" style={{ fontSize: 11 }}>RL Training Feed</h3>
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full" style={{ background: rlStep > 0 ? "#34d399" : "rgba(255,255,255,0.2)", animation: rlStep > 0 ? "shimmer 1.5s infinite" : "none" }} />
                <span className="text-white/25 font-mono" style={{ fontSize: 9 }}>{rlStep} steps</span>
              </div>
            </div>
            <div className="space-y-1" style={{ maxHeight: 80, overflowY: "auto" }}>
              {interactions.slice(0, 6).map((item, i) => (
                <div key={i} className="flex items-center gap-2 fade-in">
                  <span style={{ fontSize: 9, color: item.action === "click" ? "#34d399" : "#f87171" }}>{item.action === "click" ? "▲ +reward" : "▼ −reward"}</span>
                  <span className="text-white/30 font-mono flex-1 truncate" style={{ fontSize: 9 }}>{item.title}</span>
                  <span className="text-white/15 font-mono" style={{ fontSize: 8 }}>step {rlStep - i}</span>
                </div>
              ))}
              {interactions.length === 0 && <p className="text-white/20" style={{ fontSize: 10 }}>No interactions yet — click or skip items to train</p>}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-5 py-3">
            <h3 className="text-white/40 mb-3" style={{ fontSize: 11 }}>Interaction History</h3>
            {interactions.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-32 text-center">
                <div style={{ fontSize: 28, opacity: 0.15 }}>🤖</div>
                <p className="text-white/20 mt-2" style={{ fontSize: 11 }}>Interact with results to train the DQN</p>
              </div>
            ) : interactions.map((item, i) => (
              <div key={i} className="flex items-center gap-2.5 py-1.5 fade-in" style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <BookCover color={item.cover_color} title={item.title} size="sm" />
                <div className="flex-1 min-w-0">
                  <p className="text-white/65 truncate" style={{ fontFamily: "'Playfair Display', serif", fontSize: 11 }}>{item.title}</p>
                  <p className="text-white/25" style={{ fontSize: 9 }}>{item.author}</p>
                </div>
                <div className="flex flex-col items-end gap-0.5">
                  <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 99, background: item.action === "click" ? "rgba(52,211,153,0.12)" : "rgba(248,113,113,0.12)", color: item.action === "click" ? "#6ee7b7" : "#fca5a5", border: `1px solid ${item.action === "click" ? "rgba(52,211,153,0.25)" : "rgba(248,113,113,0.25)"}` }}>{item.action}</span>
                  <span className="text-white/15 font-mono" style={{ fontSize: 8 }}>{new Date(item.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## 6. `api.py` — Full Source

```python
"""
NBA System — FastAPI Backend
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import base64, io

app = FastAPI(title="NBA Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        # TODO: Replace with real imports from src/
        # from src.retrieval.search import MultimodalSearchPipeline
        # from src.rl.dqn_agent import DQNAgent
        # from src.profile.user_profile import UserProfileManager
        # _pipeline = {
        #     "search": MultimodalSearchPipeline(...),
        #     "rl": DQNAgent(...),
        #     "profiles": UserProfileManager(),
        # }
        _pipeline = {"mock": True}
    return _pipeline


class SearchRequest(BaseModel):
    query: str = ""
    image_base64: Optional[str] = None
    top_k: int = 10

class InteractRequest(BaseModel):
    user_id: str
    item_id: str
    action: str  # "click" | "skip"


def decode_image(b64: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    """Mode 1 — BLaIR text search + CLIP image search + RRF fusion."""
    p = get_pipeline()

    # TODO: Replace mock block below with real pipeline calls
    # text_results = p["search"].search_text(req.query, top_k=req.top_k * 2)
    # if req.image_base64:
    #     img_arr = decode_image(req.image_base64)
    #     img_results = p["search"].search_image(img_arr, top_k=req.top_k * 2)
    #     results = p["search"].rrf_fusion(text_results, img_results, top_k=req.top_k)
    # else:
    #     results = text_results[:req.top_k]
    # return {"results": results, "query": req.query, "total": len(results)}

    results = [
        {"id": f"b00{i}", "title": f"Mock Book {i}", "author": "Mock Author",
         "genre": "Dark Fantasy", "score": round(0.95 - i * 0.03, 2),
         "text_sim": 0.91, "img_sim": 0.88, "cover_color": "#1a0a2e"}
        for i in range(req.top_k)
    ]
    return {"results": results, "query": req.query, "total": len(results)}


@app.get("/recommend")
def recommend(user_id: str):
    """Mode 2 — 3-layer funnel: Cleora → Veto → DQN."""
    p = get_pipeline()

    # TODO: Replace mock block below with real pipeline calls
    # profile = p["profiles"].get_or_create(user_id)
    # if not profile.recent_items:
    #     return {"recommendations": p["search"].get_popular(top_k=5), "mode": "cold_start"}
    # candidates = p["search"].cleora_neighbors(profile.recent_items[-5:], top_k=50)
    # candidates = [c for c in candidates if not (c["text_sim"] < 0.3 and c["img_sim"] < 0.3)]
    # state = profile.to_state_vector()
    # scored = sorted(candidates, key=lambda c: p["rl"].q_value(state, c["action_vec"]), reverse=True)
    # return {"recommendations": scored[:5], "user_id": user_id}

    layers = ["Cleora + BLaIR", "Cleora + CLIP", "RL-DQN"]
    recs = [
        {"id": f"r00{i}", "title": f"Recommended Book {i}", "author": "Rec Author",
         "genre": "Epic Fantasy", "score": round(0.92 - i * 0.04, 2),
         "cover_color": "#0d1b2a", "layer": layers[i % 3]}
        for i in range(3)
    ]
    return {"recommendations": recs, "user_id": user_id, "funnel_size": 50}


@app.post("/interact")
def interact(req: InteractRequest):
    """Record click/skip → update profile embedding → DQN train_step."""
    p = get_pipeline()
    reward = 1.0 if req.action == "click" else -0.2

    # TODO: Replace mock block below with real pipeline calls
    # profile = p["profiles"].get_or_create(req.user_id)
    # item_vec = p["search"].get_item_vector(req.item_id)
    # profile.update(req.item_id, item_vec, action=req.action)
    # state = profile.to_state_vector()
    # p["rl"].train_step(state=state, action_id=req.item_id, reward=reward)

    return {"status": "ok", "user_id": req.user_id, "item_id": req.item_id,
            "action": req.action, "reward": reward}


@app.get("/profile")
def get_profile(user_id: str):
    """Return live profile stats for the frontend dashboard."""

    # TODO: Replace mock block below with real pipeline calls
    # profile = p["profiles"].get(user_id)
    # return {"user_id": user_id, "interaction_count": profile.total_interactions,
    #         "ctr": profile.ctr, "rl_steps": profile.rl_steps,
    #         "profile_norm": float(np.linalg.norm(profile.aggregated_vector))}

    return {"user_id": user_id, "interaction_count": 0, "ctr": 0.0,
            "rl_steps": 0, "profile_norm": 0.0}
```

---

## 7. API Response Contracts

The frontend expects these exact shapes from the backend. Your pipeline must return data matching these schemas.

### `POST /search` → results array
```json
{
  "results": [
    {
      "id": "string",
      "title": "string",
      "author": "string",
      "genre": "string",
      "score": 0.94,
      "text_sim": 0.91,
      "img_sim": 0.88,
      "cover_color": "#hex"
    }
  ],
  "total": 10
}
```

### `GET /recommend` → recommendations array
```json
{
  "recommendations": [
    {
      "id": "string",
      "title": "string",
      "author": "string",
      "score": 0.91,
      "cover_color": "#hex",
      "layer": "Cleora + BLaIR | Cleora + CLIP | RL-DQN"
    }
  ]
}
```

### `POST /interact` → confirmation
```json
{ "status": "ok", "reward": 1.0 }
```

> **Note on `cover_color`**: The frontend uses this hex value to generate the book cover gradient. If your metadata doesn't include a color, derive one deterministically from the item ID (e.g., `"#" + hashlib.md5(item_id.encode()).hexdigest()[:6]`).

---

## 8. Quick-Start Checklist

```
[ ] npm create vite@latest frontend -- --template react
[ ] Replace frontend/src/App.jsx with Section 5 source
[ ] Replace frontend/src/main.jsx with Section 2 snippet
[ ] Replace frontend/vite.config.js with Section 2 snippet
[ ] Create api.py at project root with Section 6 source
[ ] pip install fastapi uvicorn pillow numpy
[ ] Replace the 4 TODO stubs in api.py with real src/ calls
[ ] uvicorn api:app --reload --port 8000
[ ] cd frontend && npm run dev
[ ] Open http://localhost:5173 — toggle MOCK→LIVE in the header
```
