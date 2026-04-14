# Multi-User Concurrency — AgentPool Implementation — April 14, 2026

This document covers the diagnosis of the shared-model race condition that existed
in the single-agent DIF-SASRec setup, and the full implementation of an `AgentPool`
that gives each concurrent request its own isolated agent.

---

## Context

Since the Apr 13 integration, `PassiveRecommendationEngine` owned a single
`DIFSASRecAgent` (`self.sasrec`) that was shared across every request. The agent's
core state is `self.model` — 12.4M parameters on GPU. Two concurrent requests
interleaved at any `await` point and would silently corrupt each other's weights:

```
User A: GET /recommend
  → load_personal_weights("user_A")    ← model is now user_A's weights
  → await profile_manager.get_profile  ← yields control here
                                          ↕ context switch
User B: GET /recommend
  → load_personal_weights("user_B")    ← model is now user_B's weights
                                          ↕ context switch
User A: → recommend_for_user()
  → get_candidate_scores()             ← scores with user_B's weights (silent bug)
```

The same race applied between `/interact` (training, modifies weights in-place)
and `/recommend` (inference). No exception was raised — the wrong recommendations
were simply served to the wrong user.

---

## Solution: AgentPool

Eight independent `DIFSASRecAgent` instances, each with its own copy of the model
on GPU. A request borrows one agent, does its work, and returns it. No weight state
is shared between concurrent requests.

**VRAM budget:**
- Model weights: 12.4M params × 4 bytes = ~49 MB per agent
- AdamW optimizer (2 momentum buffers): ~99 MB per agent
- Total per agent: ~148 MB
- 8 agents: ~1.18 GB — well within the available 8 GB headroom

---

## How it works

### Startup (`lifespan.py`)

```
lifespan.py
  → AgentPool(n=8, retriever, cat_encoder, pretrained_path)
  → await pool.warmup()
        pool._build_agents()  ← runs in thread via run_in_executor
            for each of 8 agents:
                DIFSASRecAgent.__init__()
                loads dif_sasrec_pretrained.pt (12.4M params) onto GPU
                snapshots pretrained state into CPU RAM (_pretrained_state)
                appends agent to list
        back on event loop: put_nowait() each agent into asyncio.Queue
```

The GPU weight loads (blocking) happen in a thread pool so the event loop stays
free during startup. The `asyncio.Queue` is only populated back on the event loop
thread after all agents are built — keeping queue operations thread-safe.

Each agent independently loads the same pretrained checkpoint from disk and caches
a deep copy of the pretrained weights in CPU RAM. This snapshot is used to reset
the agent to a clean baseline when it serves a new user with no personal checkpoint.

### Request flow (`GET /recommend`)

```
request arrives
  → async with container.agent_pool.borrow() as agent:
        agent is removed from the asyncio.Queue
        (if pool is empty, request awaits here until one is returned)

        agent.load_user(user_id, data_dir)
          → if personal checkpoint exists: load it from disk
          → else: reset model to _pretrained_state (CPU → GPU copy)

        recommend_engine.recommend_for_user(user_id, agent, top_k=5)
          → Pipeline A (Cleora) — no agent involvement
          → Pipeline B: agent.get_candidate_scores(asins, cat_ids, candidates)

  → agent released back to pool via finally block (always, even on exception)
```

### Request flow (`POST /interact`)

```
request arrives
  → profile update logged (outside pool — no agent needed)

  → async with container.agent_pool.borrow() as agent:
        agent.load_user(user_id, data_dir)
          → load personal checkpoint, or reset to pretrained for new users

        recommend_engine.train_personal(user_id, item_asin, agent, click_seq_before)
          → agent.train_step() — single gradient step, modifies this agent's weights only

        agent.save_user(user_id, data_dir)
          → saves to data/profiles/<user_id>_dif_sasrec.pt

  → agent released back to pool
```

### Dirty-agent problem and the pretrained snapshot

A pool introduces a correctness risk: after User A uses an agent, that agent's
weights are User A's. If User B (a new user with no personal checkpoint) picks it
up next, the old code (`load_personal_weights`) would skip loading entirely because
no file exists — User B would then receive inference on User A's weights, silently.

The fix is the **pretrained snapshot**: at the end of `DIFSASRecAgent.__init__()`,
after the pretrained checkpoint is loaded, a `copy.deepcopy` of the model state,
optimizer state, step counter, and loss history is saved to CPU RAM. `load_user()`
always either loads the personal file or resets to this snapshot — there is no
skip path.

```python
# DIFSASRecAgent.__init__ (end)
self._pretrained_state        = copy.deepcopy(self.model.state_dict())
self._pretrained_opt_state    = copy.deepcopy(self.optimizer.state_dict())
self._pretrained_step         = self._step
self._pretrained_loss_history = list(self.loss_history)

# load_user()
if os.path.exists(path):
    self.load(path)         # personal weights
else:
    self.model.load_state_dict(self._pretrained_state)   # always clean
    self.optimizer.load_state_dict(self._pretrained_opt_state)
    self._step        = self._pretrained_step
    self.loss_history = list(self._pretrained_loss_history)
    self.model.eval()
```

CPU RAM cost of snapshots: ~49 MB × 8 agents = ~392 MB (system RAM, not VRAM).

### Overflow behaviour

`asyncio.Queue.get()` blocks naturally. If all 8 agents are busy and a 9th request
arrives, it waits until one is released. No error, no dropped request — just a
brief queue wait. Under the target concurrency of ≤10 simultaneous users, slot
exhaustion is momentary at worst.

---

## Changes

### `app/services/dif_sasrec.py`

- Added `import copy`
- End of `__init__`: snapshot `_pretrained_state`, `_pretrained_opt_state`,
  `_pretrained_step`, `_pretrained_loss_history` into CPU RAM
- Added `load_user(user_id, data_dir)` — load personal or reset to pretrained
- Added `save_user(user_id, data_dir)` — save personal checkpoint
- Added `_user_path(data_dir, user_id)` static method — path construction
  (previously duplicated in `PassiveRecommendationEngine._sasrec_path`)

### `app/services/agent_pool.py` *(new)*

| Symbol | Role |
| :--- | :--- |
| `AGENT_POOL_SIZE = 8` | Pool size constant |
| `AgentPool.__init__` | Stores config; pool not yet populated |
| `AgentPool._build_agents()` | Synchronous — creates N agents, returns list; safe to call from thread |
| `AgentPool.warmup()` | `async` — runs `_build_agents` in executor, then populates queue on event loop |
| `AgentPool.acquire()` | `async` — blocks until an agent is available |
| `AgentPool.release(agent)` | Returns agent to pool |
| `AgentPool.borrow()` | `@asynccontextmanager` — acquire + guaranteed release |
| `AgentPool.available` | Property: current idle agent count |

**Thread-safety note:** `asyncio.Queue` is not thread-safe. The original design
called `put_nowait()` from inside `run_in_executor`, which is technically unsafe
even though it would not crash in practice during startup (no concurrent getters).
The fix splits warmup into two phases: `_build_agents()` does the blocking GPU
work in a thread and returns a plain list; `warmup()` (async, on the event loop)
then populates the queue via `put_nowait()` — correct thread ownership.

### `app/services/passive_recommend.py`

- Removed `self.sasrec` — engine no longer owns any model
- Removed `_sasrec_path`, `load_personal_weights`, `save_personal_weights`
- `recommend_for_user`, `_personal_recommend`, `train_personal`, `rrf_fusion`
  all take `agent` as an explicit parameter
- Removed `import os` and `from app.services.dif_sasrec import DIFSASRecAgent`
  (no longer needed at engine level)

### `app/core/container.py`

- Added `agent_pool: Any = None` field

### `app/core/lifespan.py`

- Added `from app.services.agent_pool import AgentPool, AGENT_POOL_SIZE`
- After `recommend_engine` is built: creates `AgentPool`, `await`s `pool.warmup()`
  directly (warmup is now async and manages its own executor internally), stores
  as `container.agent_pool`

### `app/api/routes/recommend.py`

- `GET /recommend`: warm path wraps DIF-SASRec work in `async with container.agent_pool.borrow() as agent`
- `GET /rl_metrics`: borrows an agent, calls `agent.load_user()`, reads metrics, releases

### `app/api/routes/interact.py`

- `POST /interact`: training block wraps in `async with container.agent_pool.borrow() as agent`
  with `agent.load_user()` before training and `agent.save_user()` after

---

## What is NOT changed

- Pipeline A ("People Also Buy") — Cleora + content veto — no agent involvement,
  unaffected
- Cold-start path — returns random catalogue items before pool is touched
- Profile manager, Redis logging, metadata enrichment — all unchanged
- `DIFSASRecAgent.train_step`, `get_candidate_scores`, `save`, `load` — internal
  logic unchanged; only the ownership and lifecycle moved to the pool

---

## Concurrency guarantee after this change

| Scenario | Before | After |
| :--- | :--- | :--- |
| 2 concurrent `/recommend` | Silent weight contamination | Isolated — separate agents |
| `/interact` + `/recommend` simultaneously | Training overwrites inference weights | Isolated — separate agents |
| New user served by a previously-used agent | Inherits previous user's weights | Reset to pretrained baseline |
| Pool exhausted (> 8 concurrent) | N/A | 9th request awaits; no error |

---

*Report generated April 14, 2026. Branch: `Multiple_handling`.*
