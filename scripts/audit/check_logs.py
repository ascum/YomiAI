import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import json
from datetime import datetime

async def check_logs():
    MONGO_URL = "mongodb://localhost:27017"
    DB_NAME = "nba_logs"
    
    print(f"--- Checking MongoDB: {DB_NAME} ---")
    try:
        client = AsyncIOMotorClient(MONGO_URL)
        db = client[DB_NAME]
        collection = db["interactions"]
        
        # Count total
        count = await collection.count_documents({})
        print(f"Total interactions logged: {count}")
        
        # Get last 5
        cursor = collection.find().sort("timestamp", -1).limit(5)
        print("\nLast 5 interactions:")
        async for doc in cursor:
            # Clean up ObjectId for printing
            doc["_id"] = str(doc["_id"])
            print(json.dumps(doc, indent=2))
            
        if count == 0:
            print("\n💡 Tip: Try clicking a book in the UI first (make sure api.py is running)!")

    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

if __name__ == "__main__":
    asyncio.run(check_logs())
