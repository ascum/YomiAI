"""Quick smoke test for all NBA API endpoints."""
import requests, json

BASE = "http://localhost:8000"

# /health
r = requests.get(f"{BASE}/health")
h = r.json()
print("=== /health ===")
print(json.dumps(h, indent=2))
assert h.get("reranker_live"), "ERROR: reranker_live is False"
print("[OK] reranker_live: True")

# /search (live text encoding + reranker)
r = requests.post(f"{BASE}/search", json={"query": "dark fantasy magic systems", "top_k": 3})
d = r.json()
print(f"\n=== /search (live_encoding={d.get('live_encoding')}) — {len(d['results'])} results ===")
for x in d["results"]:
    reranker_score = x.get("reranker_score", "N/A")
    desc_snip = (x.get("description") or "")[:60]
    print(f"  [{x['id']}] {x['title']} by {x['author']}")
    print(f"       score={x['score']:.4f}  reranker_score={reranker_score}  desc={desc_snip!r}")

if d["results"]:
    assert "reranker_score" in d["results"][0], "ERROR: reranker_score missing from search results"
    print("[OK] reranker_score present in results")
    has_desc = any(x.get("description") for x in d["results"])
    if has_desc:
        print("[OK] description field non-empty in at least one result")
    else:
        print("[WARN] description empty in all results — parquet rebuild may not have completed yet")


# /recommend (cold start)
r = requests.get(f"{BASE}/recommend?user_id=smoke_user")
rec = r.json()
print(f"\n=== /recommend mode={rec['mode']} — {len(rec['recommendations'])} recs ===")
for x in rec["recommendations"]:
    print(f"  [{x['layer']}] {x['title']} by {x['author']}  score={x['score']:.4f}")

# /interact (click)
if rec["recommendations"]:
    item_id = rec["recommendations"][0]["id"]
    for _ in range(6):   # simulate 6 clicks to pass cold-start threshold
        r = requests.post(f"{BASE}/interact",
            json={"user_id": "smoke_user", "item_id": item_id, "action": "click"})
        print(f"\n/interact click -> {r.json()}")

# /recommend again (should now be personalized or still cold_start depending on threshold)
r = requests.get(f"{BASE}/recommend?user_id=smoke_user")
rec = r.json()
print(f"\n=== /recommend (after clicks) mode={rec['mode']} — {len(rec['recommendations'])} recs ===")

# /profile
r = requests.get(f"{BASE}/profile?user_id=smoke_user")
print(f"\n=== /profile ===\n{json.dumps(r.json(), indent=2)}")
