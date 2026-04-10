const API_BASE = "http://localhost:8000";

export const api = {
  async search(query, imageBase64 = null, sessionId = null) {
    const res = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, image_base64: imageBase64, top_k: 20, session_id: sessionId }),
    });
    return res.json();
  },

  async recommend(userId, sessionId = null) {
    const res = await fetch(`${API_BASE}/recommend?user_id=${userId}&session_id=${sessionId}`);
    return res.json();
  },

  async interact(userId, itemId, action, sessionId = null) {
    const res = await fetch(`${API_BASE}/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, item_id: itemId, action, session_id: sessionId }),
    });
    return res.json();
  },

  async profile(userId) {
    const res = await fetch(`${API_BASE}/profile?user_id=${userId}`);
    return res.json();
  },

  async rlMetrics(userId) {
    const res = await fetch(`${API_BASE}/rl_metrics?user_id=${userId}`);
    return res.json();
  },

  async askLLM(title, author, userPrompt) {
    const res = await fetch(`${API_BASE}/ask_llm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: "preview", title, author, user_prompt: userPrompt }),
    });
    return res.json();
  },
};
