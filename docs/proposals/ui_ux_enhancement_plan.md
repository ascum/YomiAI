# UI/UX Enhancement Plan — Login, Signals & Thesis Visualisations

*Written: April 14, 2026*

---

## Overview

Three categories of work, in priority order:

1. **Login / Sign-up gate** — identity before the main app, MongoDB-backed, no password
2. **Engagement Signals tooltips** — inline documentation on the Profile radar bars
3. **Thesis visualisation suite** — UI elements that make the ML pipeline visible to judges

All frontend changes maintain the existing aesthetic: `Cormorant Garamond` titles,
`Syne` body, `DM Mono` metrics, `k-cream / k-navy / k-accent` palette, Tailwind dark mode.

---

## Part 1 — Login / Sign-up Gate

### Motivation

Currently `userId` is hardcoded to `"user_demo_01"`. For a thesis demo with judges,
each evaluator needs their own identity so personalisation (DIF-SASRec online training,
profile weights) is isolated per person and persists across refreshes.

### User flow

```
App opens
  ↓
localStorage has saved userId?
  → Yes: skip gate, enter main app as that user
  → No:  render LoginPage (full screen)

LoginPage — Step 1 (Input)
  User types a display name
  Clicks "Enter"
  → POST /auth/check { username }

    Case A — user found:
      → save userId to localStorage
      → onLogin(userId) → main app renders

    Case B — user not found:
      → LoginPage transitions to Step 2 (Confirm)

LoginPage — Step 2 (Confirm new account)
  "No account found for '[name]'. Create one?"
  [ Create Account ]   [ Try Again ]

  "Create Account" → POST /auth/create { username }
                   → save userId to localStorage
                   → onLogin(userId) → main app

  "Try Again"      → back to Step 1, input cleared
```

A "Continue as Guest" link is always available — generates a `guest_<random>` ID
without hitting the database, for quick evaluation access.

### Backend — `app/api/routes/auth.py` (new)

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/auth/check` | POST | `{ username: str }` | `{ found: bool, user_id: str, display_name: str }` |
| `/auth/create` | POST | `{ username: str }` | `{ user_id: str, display_name: str }` |

**username → user_id mapping:** sanitize to lowercase alphanumeric + underscores,
prefix with `u_` to distinguish from legacy `user_demo_01` and `guest_*` IDs.
Example: `"Judge 1"` → `u_judge_1`.

**MongoDB check:** query the existing profiles collection by `user_id`.
`/auth/create` calls `profile_manager.get_profile(user_id)` which already creates
a blank profile on first access — no new collection needed.

### Frontend — new files

| File | Role |
|---|---|
| `frontend/src/components/LoginPage.jsx` | Full-screen login/signup gate |
| `frontend/src/services/api.js` | Add `api.authCheck()` and `api.authCreate()` |

**LoginPage internal states:** `idle → checking → not_found → creating → done`

The card morphs between states with a smooth CSS transition — no page navigation.
Same card, same aesthetic, different content.

**App.jsx changes:**
- `userId` initial state reads from `localStorage("yomiai_user_id")` first
- If truthy, skip gate entirely
- If null, render `<LoginPage onLogin={id => { setUserId(id); ... }} />`
- Logout: clear localStorage key, reload

---

## Part 2 — Engagement Signals Tooltips

### Motivation

The four bars (CTR, Depth, Model Fit, Diversity) are meaningful metrics but
opaque to a judge who hasn't read the documentation. Inline tooltips make the
profile panel self-explanatory during a live demo.

### Implementation

New reusable component `frontend/src/components/ui/InfoTooltip.jsx`.
A small `ⓘ` icon next to each bar label. On hover, a tooltip appears with:
- Plain-English description of what the metric means
- The formula used to compute it

**Tooltip content per bar:**

| Bar | Description | Formula |
|---|---|---|
| CTR | Ratio of positive interactions (clicks + carts) out of all session interactions | `(clicks + carts) / total` |
| Depth | How deep into the session you've gone — saturates at 20 interactions | `min(interactions / 20, 1)` |
| Model Fit | Proxy for how much positive training signal DIF-SASRec has received from you this session | `min(clicks / 10, 1) × 0.8 + 0.1` |
| Diversity | Fraction of interactions that were unique books — measures exploration vs revisiting | `unique items / total` |

Tooltip implemented with Tailwind `group` + `group-hover:opacity-100` — pure CSS,
no JS state, no library.

**Files changed:** `ProfileRadar.jsx` (add `<InfoTooltip>` per bar),
new `InfoTooltip.jsx`.

---

## Part 3 — Thesis Visualisation Suite

Ordered by implementation priority. All use existing API data — no new endpoints
required unless noted.

### A. Cold-start status pill
**Data:** `mode` field from `GET /recommend`

A persistent status indicator in the Recs tab header:
- `Discovery mode` (grey, static) — cold start, random catalogue items
- `DIF-SASRec active` (accent, live pulse dot) — personalised mode

On transition, animate the pill and show a toast:
*"Personalised mode — DIF-SASRec is now ranking for you"*

Directly demonstrates the cold-start problem to judges in real time.

### B. Click sequence strip
**Data:** `recent_items` from `GET /profile`

A horizontal scrollable strip above the You Might Like tab, labelled
*"DIF-SASRec input sequence →"*. Shows the last N book covers/titles in
chronological order with small click/cart/skip action badges.

Makes the transformer's input sequence visible — judges see exactly what
the model is attending to when generating recommendations.

### C. Loss chart phase annotations
**Data:** `loss_history` from `GET /rl_metrics`

The existing sparkline shows a dramatic drop from pretraining (~5.34) to
online adaptation (~0.5–1.0). This looks like instability without context.

Add:
- Vertical dashed divider between pretraining and online regimes
- `Pretraining` / `Online` phase labels
- Hover tooltip on the drop: *"Lower online loss reflects easier negatives,
  not instability — the pretrained model already ranks similar items well"*

### D. Inline training pulse after each click
**Data:** `sasrec_loss` from `POST /interact` (currently discarded)

After each click/cart interaction, immediately append the returned `sasrec_loss`
to the sparkline and flash a `↓ 0.847 trained` badge next to the chart.
Makes online learning feel alive rather than happening invisibly.

### E. Search modality breakdown
**Data:** `text_sim` and `img_sim` already on each search result

Replace the raw decimal numbers with two mini progress bars:
`📝 ████░░` and `🖼 ██░░░░` per result card.

When doing image search, the image bar dominates — directly demonstrates
multi-modal fusion to judges without them needing to read numbers.

### F. "New this refresh" badge
**Data:** local diff of previous vs current `you_might_like` list

After interactions trigger a rec reload, any ASIN not in the previous set
gets a subtle `New` badge that fades after 4 seconds.

Confirms to judges that the model is actually changing its output — not
serving static recommendations dressed as personalised ones.

### G. Genre preference mini-chart
**Data:** `recent_items` genres from `GET /profile`

In the Profile panel, compute a ranked count of genres from interaction history
and show a small bar chart: *"Top genres: Fantasy · Sci-Fi · Romance"*.

Directly demonstrates DIF-SASRec's category stream — the model is learning
genre preference shifts, and this makes the signal visible.

### H. Pipeline transparency on rec card hover
**Data:** `layer` field already on each recommendation

On hover, expand a tooltip on each rec card showing the full pipeline path:

- You Might Like: `HNSW KNN (200 candidates) → content veto → DIF-SASRec scoring`
- People Also Buy: `Cleora behavioral → content veto → similarity rank`

Makes the 3-layer NBA funnel architecture visible without cluttering the default UI.

### I. Before/after personalisation toggle
**Data:** cache first recommendation set in component state at session start

After 5+ interactions, show a toggle button in the Recs tab:
*"Compare: Initial vs Now"*. Renders the two lists side-by-side or as a diff.

The most powerful demo moment — judges see the same system recommending
completely different books after learning from their clicks.

---

## Implementation Order

| Phase | Items | Status |
|---|---|---|
| **Phase 1** | Login/signup gate (Part 1) | Pending |
| **Phase 1** | Engagement Signals tooltips (Part 2) | Pending |
| **Phase 2** | A — Cold-start pill, B — Sequence strip, D — Training pulse | Pending |
| **Phase 3** | C — Loss annotations, E — Modality bars, F — New badge | Pending |
| **Phase 4** | G — Genre chart, H — Pipeline hover, I — Before/after | Pending |

---

## Files to create / modify (full list)

| File | Action |
|---|---|
| `app/api/routes/auth.py` | Create |
| `app/main.py` | Register auth router |
| `frontend/src/components/LoginPage.jsx` | Create |
| `frontend/src/components/ui/InfoTooltip.jsx` | Create |
| `frontend/src/services/api.js` | Add authCheck, authCreate |
| `frontend/src/App.jsx` | localStorage userId init, LoginPage gate |
| `frontend/src/components/features/profile/ProfileRadar.jsx` | Add InfoTooltip per bar |
