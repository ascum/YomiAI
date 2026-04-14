"""
app/services/agent_pool.py — Pool of DIFSASRecAgent instances for concurrent use.

Each agent in the pool is an independent object with its own model weights on GPU.
Routes borrow an agent via `async with pool.borrow() as agent`, load the user's
personal weights, run inference or training, save, then the agent is returned to
the pool for the next request.

This eliminates the shared-model race condition that exists when a single agent
is reused across concurrent requests.

Pool size: 8 agents (configurable via AGENT_POOL_SIZE).
  - 8 agents × ~148 MB (weights + AdamW moments) ≈ 1.18 GB VRAM
  - If all slots are busy, the 9th request awaits via asyncio.Queue — no error,
    no dropped request, just a brief wait.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

log = logging.getLogger("nba_api")

AGENT_POOL_SIZE = 8


class AgentPool:
    """
    Fixed-size pool of DIFSASRecAgent instances.

    Thread-safety model: asyncio single-event-loop. acquire/release are
    non-blocking hot paths; agents are isolated objects so no shared state.
    """

    def __init__(self, n: int, retriever, category_encoder, pretrained_path: str | None):
        self._n                 = n
        self._retriever         = retriever
        self._category_encoder  = category_encoder
        self._pretrained_path   = pretrained_path
        self._pool: asyncio.Queue = asyncio.Queue()

    def _build_agents(self) -> list:
        """
        Synchronously instantiate N agents and return them as a list.

        Run in a thread via run_in_executor — the GPU weight loads are blocking
        and would stall the event loop. Returns the list so the caller can
        populate the asyncio.Queue from the event loop thread (thread-safe).
        """
        from app.services.dif_sasrec import DIFSASRecAgent

        agents = []
        for i in range(self._n):
            agent = DIFSASRecAgent(
                self._retriever,
                self._category_encoder,
                pretrained_path=self._pretrained_path,
            )
            agents.append(agent)
            log.info(f"[AgentPool] Agent {i + 1}/{self._n} ready "
                     f"(step={agent._step}, loss_history={len(agent.loss_history)} pts)")
        return agents

    async def warmup(self):
        """
        Build all agents in a thread, then populate the asyncio.Queue from the
        event loop. Keeps GPU-load blocking off the event loop while ensuring
        queue operations happen on the correct thread.
        """
        import asyncio
        agents = await asyncio.get_event_loop().run_in_executor(None, self._build_agents)
        for agent in agents:
            self._pool.put_nowait(agent)
        log.info(f"[AgentPool] Pool ready — {self._n} agents, "
                 f"pretrained_path={self._pretrained_path}")

    async def acquire(self):
        """Block until an agent is available, then remove it from the pool."""
        return await self._pool.get()

    def release(self, agent):
        """Return agent to the pool after use."""
        self._pool.put_nowait(agent)

    @asynccontextmanager
    async def borrow(self):
        """
        Context manager: acquire → yield → always release (even on exception).

        Usage:
            async with container.agent_pool.borrow() as agent:
                agent.load_user(user_id, data_dir)
                ...
        """
        agent = await self.acquire()
        try:
            yield agent
        finally:
            self.release(agent)

    @property
    def available(self) -> int:
        """Current number of idle agents (snapshot, not guaranteed stable)."""
        return self._pool.qsize()
