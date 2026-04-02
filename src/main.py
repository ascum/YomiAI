import os
import random
import numpy as np
import pandas as pd
from config import *
from retriever import Retriever
from user_profile_manager import UserProfileManager
from active_search_engine import ActiveSearchEngine
from passive_recommendation_engine import PassiveRecommendationEngine
from environment import get_reward
from evaluator import Evaluator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("Initializing Dual-Mode NBA System...")

# 1. Load Data & Components
try:
    cleora_path = os.path.join(DATA_DIR, "cleora_embeddings.npz")
    cleora_data = np.load(cleora_path)
except FileNotFoundError:
    print(f"Cleora embeddings not found! Run src/run_cleora.py first.")
    exit(1)

retriever = Retriever(DATA_DIR, cleora_data)
profile_manager = UserProfileManager(retriever)
search_engine = ActiveSearchEngine(retriever, profile_manager)
recommend_engine = PassiveRecommendationEngine(retriever, profile_manager)
evaluator = Evaluator()

# High-precision pool for query simulation
query_pool = [asin for asin in retriever.cleora_asins if asin in retriever.asin_to_idx]

# 2. Simulation Loop (Single User Journey)
USER_ID = "user_sim_001"
print(f"Starting simulation for user {USER_ID}...")

successful_actions = 0
mode_stats = {"search": 0, "recommend": 0}

while successful_actions < STEPS:
    profile = profile_manager.get_profile(USER_ID)
    current_clicks = len(profile.clicks)
    
    # Detailed logging for the first few steps to show the logic
    verbose = successful_actions < 10
    
    if current_clicks < COLD_START_THRESHOLD:
        mode = "search"
    else:
        mode = "recommend" if random.random() > 0.3 else "search"

    if verbose:
        print(f"\n--- [ACTION {successful_actions + 1}] Mode: {mode.upper()} ---")
        if profile.text_profile is not None:
            print(f"Profile Status: Warm (Clicks: {current_clicks})")
        else:
            print(f"Profile Status: Cold (New User)")

    if mode == "search":
        query_asin = random.choice(query_pool)
        q_idx = retriever.asin_to_idx[query_asin]
        q_blair = retriever.blair_index.reconstruct(q_idx)
        q_clip = retriever.clip_index.reconstruct(q_idx)
        
        results = search_engine.search(USER_ID, text_query_vec=q_blair, image_query_vec=q_clip)
        source = "search"
        if verbose:
            print(f"Active Search: User queried with item {query_asin}")
            print(f"RRF Results: Found {len(results)} multimodal matches")
    else:
        results = recommend_engine.recommend_for_user(USER_ID)
        source = "recommendation"
        if verbose:
            print(f"Passive Rec: Querying Cleora with last items: {list(profile.recent_interactions)[-3:]}")
            print(f"Personalization: RL Agent scoring {BEHAVIORAL_CANDIDATES} behavioral candidates...")

    if not results:
        query_asin = random.choice(query_pool)
        results = [(query_asin, 1.0)]
        source = "fallback"

    chosen_asin, _ = results[0]
    chosen_vec = retriever.get_asin_vec(chosen_asin)
    reward = get_reward(chosen_asin, chosen_vec)
    
    if verbose:
        print(f"System Choice: Recommended {chosen_asin}")
        print(f"User Response: {'CLICK (Reward=1)' if reward == 1 else 'SKIP (Reward=0)'}")

    if reward == 1:
        profile_manager.log_click(USER_ID, chosen_asin, source=source)
        if verbose:
            print(f"Profile Update: Aggregated embeddings re-computed for {USER_ID}")
    
    if profile.text_profile is not None:
        loss = recommend_engine.train_rl(profile, chosen_asin, reward)
        if verbose:
            print(f"RL Training: DQN updated with reward {reward} (Loss: {loss:.6f})")
    
    evaluator.log(reward)
    successful_actions += 1
    mode_stats[mode if mode in mode_stats else "search"] += 1

    if successful_actions % 100 == 0:
        print(f"[{successful_actions}/{STEPS}] Mode: {mode}, CTR: {evaluator.ctr():.4f}, Profile Size: {len(profile.clicks)}")

print("\n--- Final Dual-Mode Evaluation ---")
print(f"Total Actions: {successful_actions}")
print(f"Mode Distribution: {mode_stats}")
print(f"Final User Profile Size: {len(profile_manager.get_profile(USER_ID).clicks)} clicks")
print(f"Final Stable CTR: {evaluator.ctr():.4f}")
