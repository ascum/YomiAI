import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from redis import asyncio as aioredis

log = logging.getLogger("nba_api")

# ─── Configuration ────────────────────────────────────────────────────────────
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DB_NAME   = "nba_logs"

# ─── Connection Holders ───────────────────────────────────────────────────────
class Database:
    client: AsyncIOMotorClient = None
    db = None
    redis: aioredis.Redis = None

    @classmethod
    async def connect(cls):
        """Initialize async connections to Mongo and Redis."""
        try:
            log.info(f"Connecting to MongoDB at {MONGO_URL}...")
            cls.client = AsyncIOMotorClient(MONGO_URL)
            cls.db = cls.client[DB_NAME]
            
            # Ping check
            await cls.client.admin.command('ping')
            log.info("MongoDB connected ✓")

            log.info(f"Connecting to Redis at {REDIS_URL}...")
            cls.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await cls.redis.ping()
            log.info("Redis connected ✓")

        except Exception as e:
            log.error(f"Infrastructure connection failed: {e}")
            log.warning("System will run without persistent logging fallback.")

    @classmethod
    async def disconnect(cls):
        """Close connections gracefully."""
        if cls.client:
            cls.client.close()
        if cls.redis:
            await cls.redis.close()
        log.info("Infrastructure connections closed.")

    @classmethod
    async def log_interaction(cls, interaction_data: dict):
        """
        Push interaction to MongoDB. 
        Usually called from a background task to avoid blocking the user.
        """
        if cls.db is None:
            return
        
        try:
            collection = cls.db["interactions"]
            await collection.insert_one(interaction_data)
        except Exception as e:
            log.error(f"Failed to write to MongoDB: {e}")

db = Database()
