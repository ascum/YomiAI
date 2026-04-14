import { useState } from "react";
import { api } from "../services/api";

/**
 * LoginPage — full-screen identity gate before the main app.
 *
 * States: idle → checking → not_found → creating → done
 *
 * "done" calls onLogin(userId) and is handled by App.jsx.
 */
export function LoginPage({ onLogin, isDark, onToggleDark }) {
  const [step, setStep]         = useState("idle");   // idle | checking | not_found | creating
  const [userId, setUserId]     = useState("");
  const [error, setError]       = useState("");
  const [pendingId, setPending] = useState("");       // user_id waiting for confirm

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleCheck = async () => {
    const id = userId.trim();
    if (!id) { setError("Please enter a user ID."); return; }
    setError("");
    setStep("checking");
    try {
      const res = await api.authCheck(id);
      if (res.found) {
        onLogin(res.user_id);
      } else {
        setPending(res.user_id);
        setStep("not_found");
      }
    } catch {
      setError("Could not reach the server. Is the API running?");
      setStep("idle");
    }
  };

  const handleCreate = async () => {
    setStep("creating");
    try {
      const res = await api.authCreate(pendingId);
      onLogin(res.user_id);
    } catch {
      setError("Failed to create account. Please try again.");
      setStep("not_found");
    }
  };

  const handleRetry = () => {
    setUserId("");
    setPending("");
    setError("");
    setStep("idle");
  };

  const handleGuest = () => {
    onLogin("guest_" + Math.random().toString(36).slice(2, 8));
  };

  const isLoading = step === "checking" || step === "creating";

  // ── Shared style tokens ────────────────────────────────────────────────────

  const inputCls = `w-full px-4 py-3 rounded-xl text-[13px] transition-all duration-200
    bg-white dark:bg-[#fffef7]/5
    border border-[#babbbd] dark:border-[#627d9a]/60
    text-[#2e3257] dark:text-[#fffef7]
    placeholder:text-[#babbbd] dark:placeholder:text-[#627d9a]
    focus:border-[#2e3257] dark:focus:border-[#dfc5a4] focus:outline-none`;

  const primaryBtn = `w-full py-3 rounded-xl text-[13px] font-semibold tracking-wide
    transition-all duration-200 shadow-sm
    bg-[#2e3257] dark:bg-[#fffef7]
    text-[#fffef7] dark:text-[#2e3257]
    hover:bg-[#dfc5a4] hover:text-[#2e3257] dark:hover:bg-[#dfc5a4]
    disabled:opacity-40 disabled:cursor-not-allowed`;

  const ghostBtn = `w-full py-3 rounded-xl text-[13px] font-semibold tracking-wide
    transition-all duration-200
    border border-[#babbbd] dark:border-[#627d9a]/60
    text-[#627d9a] dark:text-[#627d9a]
    hover:border-[#2e3257] dark:hover:border-[#dfc5a4]
    hover:text-[#2e3257] dark:hover:text-[#dfc5a4]`;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#fffef7] dark:bg-[#1a1b2e] flex flex-col items-center justify-center px-4 transition-colors duration-300">

      {/* Dark mode toggle */}
      <button
        onClick={onToggleDark}
        className="fixed top-5 right-5 w-8 h-8 rounded-full border border-[#babbbd] dark:border-[#627d9a]/60
                   flex items-center justify-center text-[14px] text-[#627d9a]
                   hover:border-[#2e3257] dark:hover:border-[#dfc5a4] transition-all duration-200"
        title="Toggle dark mode"
      >
        {isDark ? "☀" : "☾"}
      </button>

      {/* Card */}
      <div className="w-full max-w-sm">

        {/* Header */}
        <div className="text-center mb-10">
          <h1
            className="text-[#2e3257] dark:text-[#fffef7] leading-none mb-2"
            style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 42, fontWeight: 600 }}
          >
            Yomi愛
          </h1>
          <p
            className="text-[#627d9a] dark:text-[#627d9a]"
            style={{ fontFamily: "'Syne', sans-serif", fontSize: 12, letterSpacing: "0.08em" }}
          >
            Multimodal Book Recommendation Engine
          </p>
          <div className="mt-4 h-px bg-[#babbbd]/40 dark:bg-[#627d9a]/30 mx-8" />
        </div>

        {/* ── Step: idle / checking ─────────────────────────────────────── */}
        {(step === "idle" || step === "checking") && (
          <div className="space-y-3">
            <p
              className="text-[#2e3257] dark:text-[#fffef7] mb-4"
              style={{ fontFamily: "'Syne', sans-serif", fontSize: 13 }}
            >
              Enter your user ID to begin.
            </p>

            <input
              type="text"
              className={inputCls}
              placeholder="e.g. user_demo_01"
              value={userId}
              onChange={e => { setUserId(e.target.value); setError(""); }}
              onKeyDown={e => e.key === "Enter" && !isLoading && handleCheck()}
              disabled={isLoading}
              autoFocus
              style={{ fontFamily: "'DM Mono', monospace" }}
            />

            {error && (
              <p className="text-[11px] text-rose-400" style={{ fontFamily: "'Syne', sans-serif" }}>
                {error}
              </p>
            )}

            <button
              className={primaryBtn}
              onClick={handleCheck}
              disabled={isLoading || !userId.trim()}
              style={{ fontFamily: "'Syne', sans-serif" }}
            >
              {step === "checking" ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  Checking…
                </span>
              ) : "Enter"}
            </button>

            <button
              className={ghostBtn}
              onClick={handleGuest}
              disabled={isLoading}
              style={{ fontFamily: "'Syne', sans-serif" }}
            >
              Continue as Guest
            </button>
          </div>
        )}

        {/* ── Step: not_found — confirm create or retry ─────────────────── */}
        {(step === "not_found" || step === "creating") && (
          <div className="space-y-3">
            <div className="px-4 py-3 rounded-xl bg-[#babbbd]/10 dark:bg-[#627d9a]/10 border border-[#babbbd]/40 dark:border-[#627d9a]/30">
              <p
                className="text-[#2e3257] dark:text-[#fffef7] leading-relaxed"
                style={{ fontFamily: "'Syne', sans-serif", fontSize: 13 }}
              >
                No account found for{" "}
                <span className="font-mono text-[#dfc5a4]">"{pendingId}"</span>.
                <br />Would you like to create one?
              </p>
            </div>

            {error && (
              <p className="text-[11px] text-rose-400" style={{ fontFamily: "'Syne', sans-serif" }}>
                {error}
              </p>
            )}

            <button
              className={primaryBtn}
              onClick={handleCreate}
              disabled={step === "creating"}
              style={{ fontFamily: "'Syne', sans-serif" }}
            >
              {step === "creating" ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  Creating…
                </span>
              ) : "Create Account"}
            </button>

            <button
              className={ghostBtn}
              onClick={handleRetry}
              disabled={step === "creating"}
              style={{ fontFamily: "'Syne', sans-serif" }}
            >
              Try a different ID
            </button>
          </div>
        )}

        {/* Footer */}
        <p
          className="text-center mt-10 text-[#babbbd] dark:text-[#627d9a]/60"
          style={{ fontFamily: "'DM Mono', monospace", fontSize: 10 }}
        >
          No password required · Thesis Demo 2026
        </p>
      </div>
    </div>
  );
}
