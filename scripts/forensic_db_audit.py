import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

async def forensic_audit(user_id="user_demo_01"):
    MONGO_URL = "mongodb://localhost:27017"
    DB_NAME = "nba_logs"
    
    print(f"🔍 === NBA System Forensic Audit: User '{user_id}' ===\n")
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]

    # 1. Check Interaction Collection (The Audit Trail)
    print("--- 1. Testing Interaction Logging ---")
    interactions = db["interactions"]
    count = await interactions.count_documents({"user_id": user_id})
    print(f"Total events found for this user: {count}")
    
    last_event = await interactions.find_one({"user_id": user_id}, sort=[("timestamp", -1)])
    if last_event:
        print(f"Latest Action: {last_event.get('action')} on ASIN {last_event.get('asin')}")
        print(f"Timestamp: {last_event.get('timestamp')}")
    else:
        print("❌ No interactions found. Please click something in the UI first!")

    # 2. Check Profiles Collection (The "Brain")
    print("\n--- 2. Testing Aggregated Profile (Stateless State) ---")
    profiles = db["profiles"]
    prof_doc = await profiles.find_one({"user_id": user_id})
    
    if prof_doc:
        print("✅ Profile document found in MongoDB.")
        print(f"History Summary Size: {len(prof_doc.get('recent_history', []))} items")
        
        # Check embeddings
        has_text = len(prof_doc.get("text_profile", [])) > 0
        has_vis  = len(prof_doc.get("visual_profile", [])) > 0
        has_cle  = len(prof_doc.get("cleora_profile", [])) > 0
        
        print(f"Text Fingerprint: {'✅ Present' if has_text else '❌ Missing'}")
        print(f"Visual Fingerprint: {'✅ Present' if has_vis else '❌ Missing'}")
        print(f"Behavioral Fingerprint (Cleora): {'✅ Present' if has_cle else '❌ Missing'}")
        
        # Check new summary fields
        search_count = len(prof_doc.get("recent_searches", []))
        rec_count = len(prof_doc.get("recent_recs", []))
        print(f"Recent Searches Persisted: {search_count}")
        print(f"Recent Recommendations Persisted: {rec_count}")
    else:
        print("❌ Profile not found in MongoDB. Migration might not have triggered yet.")

    # 3. Check for leftover JSON (Migration Check)
    print("\n--- 3. Local Migration Status ---")
    import os
    json_path = f"src/data/profiles/{user_id}.json"
    if os.path.exists(json_path):
        print(f"⚠️  Note: Local JSON file still exists at '{json_path}'.")
        print("   (The system is now using MongoDB, but I haven't auto-deleted the JSON for safety).")
    else:
        print("✅ Local JSON has been removed (or never existed). Clean stateless operation.")

    print("\n=== Audit Complete ===")

if __name__ == "__main__":
    import sys
    uid = sys.argv[1] if len(sys.argv) > 1 else "user_demo_01"
    asyncio.run(forensic_audit(uid))
