"""
app/infrastructure/database.py — MongoDB + Redis connection lifecycle.

Moved from src/database.py. Logic unchanged; import path updated.
"""
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from redis import asyncio as aioredis

log = logging.getLogger("nba_api")

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DB_NAME   = "nba_logs"


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
            await cls.client.admin.command("ping")
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
        """Push interaction to the immutable 'interactions' collection."""
        if cls.db is None:
            return
        try:
            await cls.db["interactions"].insert_one(interaction_data)
        except Exception as e:
            log.error(f"Failed to write interaction to MongoDB: {e}")

    @classmethod
    async def fetch_profile(cls, user_id: str) -> dict:
        """Retrieve the latest aggregated profile from the 'profiles' collection."""
        if cls.db is None:
            return None
        try:
            return await cls.db["profiles"].find_one({"user_id": user_id})
        except Exception as e:
            log.error(f"Failed to fetch profile from MongoDB: {e}")
            return None

    @classmethod
    async def upsert_profile(cls, user_id: str, profile_data: dict):
        """Atomically update or create a user profile summary."""
        if cls.db is None:
            return
        try:
            await cls.db["profiles"].update_one(
                {"user_id": user_id},
                {"$set": profile_data},
                upsert=True,
            )
        except Exception as e:
            log.error(f"Failed to upsert profile to MongoDB: {e}")


db = Database()
