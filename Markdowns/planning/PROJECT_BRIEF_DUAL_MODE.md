# Project Brief: Dual-Mode Multimodal Recommendation System

## Executive Summary

This project develops a production-ready recommendation system that supports **two distinct interaction modes**: (1) Active Search where users query with text/image, and (2) Passive Recommendation where the system anticipates needs based on behavioral profiles. The system combines **multimodal content understanding** (BLaIR + CLIP) with **collaborative filtering** (Cleora or RL-based) to deliver personalized, contextually relevant recommendations.

## Problem Statement

Modern recommendation systems face critical challenges:

1. **Single-Mode Limitation**:
   - Traditional systems are either pure search or pure recommendation
   - Users need both: active exploration AND passive discovery
   - No seamless integration between search and recommendation

2. **User Behavior Disconnection**:
   - Search results don't inform recommendations
   - Recommendations ignore search history
   - No continuous learning from user interactions

3. **Multimodal Query Gap**:
   - Most systems handle only text queries
   - Visual search (image upload) poorly supported
   - Cannot combine text + image in single query

4. **Static vs Adaptive**:
   - Fixed fusion weights don't adapt to query type
   - No learning from user feedback
   - Cannot evolve with changing preferences

---

## Solution: Dual-Mode Multimodal System

### Architecture Overview

Our system operates in **two complementary modes**:

**Mode 1: Active Search** (User-Initiated)
```
User Query (Text + Image) 
    ↓
BLaIR + CLIP Direct Search 
    ↓
RRF Fusion 
    ↓
Results → Update User Profile
```

**Mode 2: Passive Recommendation** (System-Initiated)
```
User Behavior Profile 
    ↓
Collaborative Filter (Cleora or RL-CF)
    ↓
BLaIR + CLIP Sanity Check 
    ↓
RRF Fusion 
    ↓
Personalized Recommendations → Update Profile
```

Both modes continuously update the **User Behavior Profile**, creating a closed feedback loop.

---

## Mode 1: Active Search

### Purpose
Enable users to actively search using text, image, or both modalities.

### Supported Query Types

**Text-Only Query**:
```
User: "mystery novels with strong female protagonists"
System: BLaIR semantic search → Top-K results
```

**Image-Only Query** (Answers teacher's question: "What if user uploads image?"):
```
User: [uploads photo of book cover]
System: CLIP visual search → Visually similar books
```

**Multimodal Query**:
```
User: "fantasy novels" + [image of dark atmospheric cover]
System: BLaIR + CLIP combined → RRF fusion → Best matches
```

### Pipeline

```python
class ActiveSearchEngine:
    def search(self, user_id, text_query=None, image_query=None, top_k=10):
        results = []
        
        # BLaIR text search
        if text_query:
            text_emb = self.blair_encoder.encode(text_query)
            blair_results = self.blair_index.search(text_emb, k=50)
            results.append(('blair', blair_results))
        
        # CLIP image search
        if image_query:
            image_emb = self.clip_encoder.encode_image(image_query)
            clip_results = self.clip_index.search(image_emb, k=50)
            results.append(('clip', clip_results))
        
        # RRF fusion
        final_ranking = self.rrf_fusion(results)
        
        # Log to user profile
        self.profile_manager.log_search(
            user_id, text_query, image_query, final_ranking[:top_k]
        )
        
        return final_ranking[:top_k]
```

---

## Mode 2: Passive Recommendation

### Purpose
Proactively recommend items based on accumulated user behavior.

### User Behavior Profile

**Components**:
```python
class UserBehaviorProfile:
    # Interaction history
    searches = []           # All search queries (text + image)
    clicks = []            # Clicked items
    purchases = []         # Purchased items
    
    # Aggregated embeddings
    text_profile = None    # Weighted avg of BLaIR embeddings
    visual_profile = None  # Weighted avg of CLIP embeddings
    
    # Behavioral patterns
    preferred_categories = Counter()
    recent_interactions = deque(maxlen=50)
    temporal_weights = []  # Recent = higher weight
```

**Profile Update Logic**:
```python
def update_profile(user_id, interaction):
    """
    Continuously update user profile from interactions
    """
    profile = get_profile(user_id)
    
    # Record interaction
    profile.interactions.append(interaction)
    
    # Recompute aggregated embeddings with temporal weighting
    all_items = [i['item_id'] for i in profile.interactions]
    weights = compute_temporal_weights(profile.interactions)  # Recent = higher
    
    item_embeddings = db.get_embeddings(all_items)
    
    # Weighted average
    profile.text_profile = np.average(
        [e['blair'] for e in item_embeddings], 
        weights=weights
    )
    profile.visual_profile = np.average(
        [e['clip'] for e in item_embeddings],
        weights=weights
    )
    
    save_profile(profile)
```

### Recommendation Pipeline

```python
class PassiveRecommendationEngine:
    def recommend_for_user(self, user_id, top_k=10):
        profile = self.profile_manager.get_profile(user_id)
        
        if not profile.has_sufficient_history():
            return self.get_trending_items(top_k)
        
        # Step 1: Collaborative Filtering
        candidates = self.collaborative_filter(profile, top_n=100)
        
        # Step 2: Content Sanity Check
        verified = self.content_verify(
            candidates,
            user_text_profile=profile.text_profile,
            user_visual_profile=profile.visual_profile
        )
        
        # Step 3: RRF Re-ranking
        final_ranking = self.rrf_fusion(verified)
        
        return final_ranking[:top_k]
```

---

## Collaborative Filtering: Cleora vs RL

### Current: Cleora-Based

**Approach**: Multi-seed retrieval from user's interaction history

```python
def cleora_collaborative_filter(user_profile, top_n=100):
    """
    Query Cleora with each item in user's history
    """
    candidates = set()
    
    for item in user_profile.recent_interactions:
        neighbors = cleora_index.search(item, k=50)
        candidates.update(neighbors)
    
    # Remove items user already has
    candidates = candidates - set(user_profile.clicks)
    
    return list(candidates)[:top_n]
```

**Pros**: Fast, deterministic, no training needed  
**Cons**: Static embeddings, no learning from feedback

### Proposed: RL-Based Collaborative Filter

**Approach**: Deep Q-Network learns user-item interaction patterns

```python
class RLCollaborativeFilter:
    def __init__(self, item_catalog):
        self.dqn = CollaborativeFilterDQN(
            state_dim=256,  # User profile embedding
            action_dim=len(item_catalog)  # Q-value per item
        )
    
    def get_candidates(self, user_profile, top_n=100):
        """
        Use DQN to score items based on user profile
        """
        state = self.encode_user_state(user_profile)
        
        with torch.no_grad():
            q_values = self.dqn(state)
        
        # Select top-N items with highest Q-values
        top_items = torch.topk(q_values, top_n).indices
        
        return [self.item_catalog[i] for i in top_items]
```

**Training**:
```python
def train_rl_cf(interaction_data):
    """
    Train with temporal holdout
    
    Reward: +1 if recommended item in user's future interactions
    """
    for user_id in interaction_data.user_ids:
        # 80/20 temporal split
        history = interaction_data.get_history(user_id, until='80%')
        future = interaction_data.get_history(user_id, from='80%')
        
        # Build profile from history
        profile = build_profile(history)
        
        # Generate candidates
        candidates = rl_cf.get_candidates(profile, top_n=50)
        
        # Compute reward
        future_items = set(future['item_id'].values)
        rewards = [1 if item in future_items else 0 for item in candidates]
        
        # Train DQN
        rl_cf.train_step(profile, candidates, rewards)
```

**Pros**: Learns from feedback, adapts over time  
**Cons**: Requires training, more complex

### Hybrid Approach (Recommended)

```python
class AdaptiveCollaborativeFilter:
    def get_candidates(self, user_profile, top_n=100):
        """
        Use RL-CF for warm users, Cleora for cold users
        """
        num_interactions = len(user_profile.clicks)
        
        if num_interactions >= 20:
            # Warm: Use RL-CF
            return self.rl_cf.get_candidates(user_profile, top_n)
        elif num_interactions >= 5:
            # Medium: Blend both
            rl_cands = self.rl_cf.get_candidates(user_profile, top_n//2)
            cleora_cands = self.cleora.get_candidates(user_profile, top_n//2)
            return list(set(rl_cands + cleora_cands))
        else:
            # Cold: Use Cleora
            return self.cleora.get_candidates(user_profile, top_n)
```

---

## Content Sanity Check Layer

### Purpose
Verify collaborative filtering candidates match user's content preferences.

### Implementation

```python
def content_verify(candidates, user_text_profile, user_visual_profile):
    """
    BLaIR + CLIP sanity check on collaborative candidates
    """
    verified = []
    
    for candidate_item in candidates:
        # Get item embeddings
        item_blair = db.get_blair_embedding(candidate_item)
        item_clip = db.get_clip_embedding(candidate_item)
        
        # Similarity to user profile
        text_similarity = cosine_similarity(user_text_profile, item_blair)
        visual_similarity = cosine_similarity(user_visual_profile, item_clip)
        
        # Hard veto: must pass threshold in at least one modality
        if text_similarity >= 0.3 or visual_similarity >= 0.3:
            verified.append({
                'item': candidate_item,
                'text_score': text_similarity,
                'visual_score': visual_similarity
            })
    
    return verified
```

---

## RRF Fusion

### Purpose
Merge BLaIR (text) and CLIP (image) rankings into unified list.

### Formula
```
RRF_score(item) = Σ 1 / (k + rank_i)
```

Where:
- k = constant (default 60)
- rank_i = position in ranking i

### Implementation

```python
def reciprocal_rank_fusion(rankings, k=60):
    """
    Merge multiple rankings using RRF
    
    Args:
        rankings: List of (modality, ranked_items) tuples
    """
    scores = defaultdict(float)
    
    for modality, ranked_items in rankings:
        for rank, item in enumerate(ranked_items):
            scores[item] += 1 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## System Flow Examples

### Example 1: New User → Active Search → Profile Building

```
1. User arrives (no profile)
   → System shows trending items

2. User searches: "science fiction" + [uploads futuristic cover image]
   → Active Search Mode:
      - BLaIR: semantic search for "science fiction"
      - CLIP: visual search for futuristic covers
      - RRF fusion
   → Results: "Dune", "Foundation", "Neuromancer"

3. User clicks "Dune"
   → Profile created:
      - text_profile = BLaIR embedding of "Dune"
      - visual_profile = CLIP embedding of Dune cover
      - preferred_categories = ['sci-fi']

4. Next visit
   → Recommendation Mode activates:
      - Cleora/RL-CF finds similar sci-fi books
      - BLaIR/CLIP verify against user profile
      - Personalized recommendations shown
```

### Example 2: Returning User → Passive Recommendations

```
1. User has 50+ interactions
   → Homepage loads Recommendation Mode:
      - RL-CF generates candidates based on learned patterns
      - BLaIR/CLIP verify against aggregated profile
      - Top-10 personalized recommendations displayed

2. User clicks recommended item
   → Profile updated with new interaction
   → Future recommendations adapt

3. User searches with new image
   → Active Search Mode handles query
   → Results logged to profile
   → Continuous learning cycle
```

---

## Dataset & Scale

### Amazon Reviews 2023
- **Post-K-Core Filtering**:
  - 100,000+ interactions
  - 1,000-10,000 users
  - 500-5,000 items

### Multimodal Embeddings
- **Source**: Hugging Face Hub (`minhkhang26/my-nba-project-data`)
- **Format**: 90 NPZ chunks
- **Embeddings**:
  - BLaIR: 1024-dim text
  - CLIP: 512-dim text + image

---

## Evaluation Strategy

### Offline Metrics (Temporal Holdout)

```python
def evaluate_system(test_data):
    """
    Evaluate both modes on held-out temporal data
    """
    metrics = {
        'search_mode': {},
        'recommendation_mode': {}
    }
    
    for user_id in test_data.user_ids:
        # Split: 80% history, 20% future
        history = test_data.get_history(user_id, until='80%')
        future = test_data.get_history(user_id, from='80%')
        
        # Build profile from history
        profile = build_profile(history)
        
        # Test Recommendation Mode
        recs = recommendation_engine.recommend_for_user(user_id, top_k=10)
        
        future_items = set(future['item_id'].values)
        hit = int(any(item in future_items for item in recs))
        
        metrics['recommendation_mode']['hit_rate'].append(hit)
        # ... other metrics
    
    return metrics
```

### Ablation Studies

**Ablation 1**: Cleora vs RL-CF
- Compare collaborative filtering approaches
- Metrics: Hit Rate, NDCG, Coverage

**Ablation 2**: With/Without Profile Updates
- Static profile vs continuous learning
- Show improvement over time

**Ablation 3**: Single-Mode vs Dual-Mode
- Search-only vs Search+Recommendation
- Demonstrate synergy

**Ablation 4**: Content Sanity Check Impact
- With/without BLaIR+CLIP verification
- Show quality improvement

---

## Innovation & Contributions

### Novel Aspects

1. **Dual-Mode Integration**: Seamless combination of active search and passive recommendation
2. **Continuous Profile Learning**: Every interaction updates behavioral profile
3. **Multimodal Query Support**: Text, image, and hybrid queries
4. **Adaptive Collaborative Filtering**: Can use Cleora (fast) or RL-CF (learned)
5. **Content Verification Layer**: Prevents hallucinated recommendations

### Research Contributions

1. **Unified Framework**: Single system handles search and recommendation
2. **Profile-Driven Personalization**: Behavioral embeddings from multimodal content
3. **RL-Based Collaborative Filtering**: Alternative to static graph embeddings
4. **Empirical Validation**: Rigorous evaluation of dual-mode approach

---

## Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Data Engineering** | 2 weeks | K-Core filtered dataset, user profiles |
| **Active Search Mode** | 2 weeks | BLaIR+CLIP search with RRF |
| **Profile Manager** | 2 weeks | Continuous profile updates |
| **Recommendation Mode (Cleora)** | 2 weeks | Collaborative filtering pipeline |
| **RL-CF Enhancement (Optional)** | 3 weeks | RL-based collaborative filter |
| **Evaluation** | 2 weeks | Ablation studies, metrics |
| **Frontend** | 2 weeks | React UI for both modes |
| **Documentation** | 2 weeks | Thesis + demo video |

**Total**: ~15-18 weeks

---

## Technical Stack

**Core Models**:
- BLaIR (text understanding)
- CLIP (image understanding)
- Cleora (behavioral embeddings)
- PyTorch DQN (optional RL-CF)

**Infrastructure**:
- Vector DB: FAISS
- Backend: FastAPI
- Frontend: React
- Database: PostgreSQL + pgvector
- Caching: Redis

---

## Answering Teacher's Concerns

| Teacher's Concern | Our Solution |
|------------------|--------------|
| "No user behavior connection" | Continuous profile updates from all interactions |
| "Cleora isolated" | Integrated into recommendation mode, queries based on user profile |
| "What if user uploads image?" | Active Search Mode handles image queries via CLIP |
| "Markov → RL progression" | Optional RL-CF replacement for Cleora |
| "Search vs Recommendation gap" | Dual-mode system unifies both |

---

## Future Work

- **Online Learning**: Real-time profile updates
- **Multi-Objective Optimization**: Balance accuracy, diversity, novelty
- **Cross-Domain Transfer**: Apply to other product categories
- **Advanced RL**: Actor-Critic methods for continuous action space
- **Explainability**: Why this recommendation?

---

**Project Status**: Revised Architecture - Dual-Mode System  
**Focus**: Active Search + Passive Recommendation with Continuous Profile Learning  
**RL Role**: Optional enhancement for collaborative filtering  
**Last Updated**: March 2026  
**Contact**: nguyeenkhanhs.cs@gmail.com, Danko Pham
