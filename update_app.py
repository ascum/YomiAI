import re

def update_app():
    with open('frontend/src/App.jsx', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add state variables for Dark Mode and Cart
    content = content.replace(
        'const [activeTab, setActiveTab] = useState("search");',
        'const [activeTab, setActiveTab] = useState("search");\n  const [isDark, setIsDark] = useState(true);\n  const [activeRightTab, setActiveRightTab] = useState("profile");\n  const [cart, setCart] = useState([]);'
    )

    # 2. Add Cart Header Button & Dark Mode Toggle
    header_controls = """{/* Cart, Theme & Mock/Live toggle */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsDark(!isDark)}
              className="px-3 py-1.5 rounded-lg transition-all duration-200"
              style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", fontSize: 13 }}
            >
              {isDark ? "☀️" : "🌙"}
            </button>
            <button
              className="px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-2"
              style={{ background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.3)", color: "#10b981", fontSize: 13, fontWeight: 600 }}
            >
              Cart <span style={{ background: "#10b981", color: "#000", padding: "0 6px", borderRadius: 10, fontSize: 10 }}>{cart.length}</span>
            </button>
          </div>
          
          <div style={{ width: 1, height: 28, background: "rgba(255,255,255,0.08)" }} />

          {/* Mock/Live */}"""
    content = content.replace('{/* Mock/Live toggle */}', header_controls)

    # 3. Handle Right Panel Tabs
    profile_panel_start = '{/* ── RIGHT PANEL — Profile + Activity ── */}'
    right_tabs = """{/* ── RIGHT PANEL — Profile + Activity ── */}
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
            <div className="flex flex-col flex-1 overflow-y-auto">"""
    
    content = content.replace(
        '{/* ── RIGHT PANEL — Profile + Activity ── */}\n        <div className="flex flex-col flex-1 overflow-hidden min-w-0">',
        right_tabs
    )
    
    # Close the profile tab div before Interaction history
    content = content.replace(
        '{/* Interaction History */}',
        '</div>\n          ) : (\n            <div className="flex-col flex-1 overflow-y-auto">\n              {/* Interaction History */}'
    )
    
    # Close the history tab div
    content = content.replace(
        '</div>\n        </div>\n      </div>\n    </div>\n  );\n}',
        '</div>\n            </div>\n          )}\n        </div>\n      </div>\n    </div>\n  );\n}'
    )

    # 4. Cart interaction fixes and HandleInteract Cart
    content = re.sub(
        r'onClick\{\(e\)\s*=>\s*\{\s*e\.stopPropagation\(\);\s*onInteract\(book,\s*"cart"\);\s*alert\(`Added.*?`\);\s*\}\}',
        'onClick={(e) => { e.stopPropagation(); onInteract(book, "cart"); }}',
        content
    )
    content = content.replace(
        'addToast(`🛒 Added "${book.title}" to Cart — STRONG positive reward`, "success");',
        'addToast(`"${book.title}" — Added to Cart`, "success");\n      setCart(prev => [...prev, book]);'
    )
    
    # 5. Remove Emojis & Update text
    content = content.replace('📖', '')
    content = content.replace('🔍 Active Search', 'Active Search')
    content = content.replace('✨ Recommendations', 'Recommendations')
    content = content.replace('✨ Ask AI', 'Ask AI')
    content = content.replace('🖼', '')
    content = content.replace('📚 Clicked', 'Clicked')
    content = content.replace('Consulting local LLM...', 'Thinking...')

    # 6. BGE Reranker Badge Display in SearchResultCard
    badge_replacement = """<ScoreBadge score={book.text_sim !== undefined ? book.text_sim : book.score} label="Text" />
            <ScoreBadge score={book.img_sim !== undefined ? book.img_sim : book.score * 0.95} label="Img" />
            {book.reranker_score !== undefined && (
               <ScoreBadge score={book.reranker_score} label="Rerank" />
            )}"""
    content = content.replace(
        '<ScoreBadge score={book.text_sim !== undefined ? book.text_sim : book.score} label="Text" />\n            <ScoreBadge score={book.img_sim !== undefined ? book.img_sim : book.score * 0.95} label="Img" />',
        badge_replacement
    )

    # 7. CSS and Theme Inject
    content = content.replace(
        '<div className="min-h-screen text-white" style={{',
        '<div className={`min-h-screen text-white ${!isDark ? "theme-light" : ""}`} style={{'
    )
    
    css_additions = """
        .theme-light {
          filter: invert(1) hue-rotate(180deg);
        }
        .theme-light img, .theme-light .book-cover-uninvert {
          filter: invert(1) hue-rotate(180deg);
        }
    """
    content = content.replace('</style>', css_additions + '</style>')
    
    # BookCover uninvert class
    content = content.replace(
        'className="rounded-sm flex-shrink-0 relative overflow-hidden"',
        'className={`rounded-sm flex-shrink-0 relative overflow-hidden ${imageUrl ? "book-cover-uninvert" : ""}`}'
    )

    with open('frontend/src/App.jsx', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    update_app()
