# NBA System Architecture: Dual-Mode Multimodal Recommendation

## Executive Summary

This system supports **two distinct interaction modes**:
1. **Active Search Mode**: User queries with text/image → Direct multimodal search
2. **Passive Recommendation Mode**: System anticipates needs based on user behavior profile

Both modes leverage BLaIR + CLIP multimodal encoders, but serve different purposes. Search results continuously update the user's behavior profile, which drives future recommendations.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                        │
│  • Active Search: Text query + Image upload                     │
│  • Passive Browse: View recommendations                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  MODE 1:        │     │  MODE 2:         │
│  ACTIVE SEARCH  │     │  RECOMMENDATION  │
└─────────────────┘     └──────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTIMODAL ENCODING LAYER (Shared)                  │
│  • BLaIR: Text semantic understanding                           │
│  • CLIP: Image visual understanding                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌───────────────────────┐
│  Direct Search   │    │  Profile-Based        │
│  (Content Only)  │    │  Retrieval            │
│                  │    │  (Behavioral + RL)    │
└──────────────────┘    └───────────────────────┘
         │                       │
         │                       ▼
         │              ┌────────────────────┐
         │              │  Cleora OR RL-CF   │
         │              │  (Collaborative)   │
         │              └────────────────────┘
         │                       │
         │                       ▼
         │              ┌────────────────────┐
         │              │  BLaIR + CLIP      │
         │              │  Sanity Check      │
         │              └────────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   RRF Fusion Layer    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Final Rankings      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Update User Profile  │
         │  (Clicks, Views, etc) │
         └───────────────────────┘
```

---

## Mode 1: Active Search (User-Initiated Query)

### Purpose
User actively searches with text and/or image to find specific products.

### Pipeline

```
User Input: "mystery novels" + [dark book cover image]
    ↓
┌─────────────────────────────────────────┐
│  Step 1: Encode Query                   │
│  • BLaIR encodes text: "mystery novels" │
│  • CLIP encodes image: dark cover       │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 2: Direct Content Search          │
│  • BLaIR: Find semantically similar     │
│  • CLIP: Find visually similar          │
│  • Both search entire catalog           │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 3: RRF Fusion                     │
│  • Merge BLaIR rankings                 │
│  • Merge CLIP rankings                  │
│  • Output: Top-K results                │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 4: Record to User Profile         │
│  • Log query (text + image)             │
│  • Track which results user clicked     │
│  • Update behavior embeddings           │
└─────────────────────────────────────────┘
```

### Implementation

```python
class ActiveSearchEngine:
    def __init__(self, blair_index, clip_index):
        self.blair_index = blair_index
        self.clip_index = clip_index
    
    def search(self, user_id, text_query=None, image_query=None, top_k=10):
        """
        User actively searches with text and/or image
        
        Args:
            text_query: User's text input (optional)
            image_query: User's uploaded image (optional)
            top_k: Number of results to return
        """
        results = []
        
        # BLaIR text search
        if text_query:
            text_embedding = self.blair_encoder.encode(text_query)
            blair_results = self.blair_index.search(text_embedding, k=50)
            results.append(('blair', blair_results))
        
        # CLIP image search
        if image_query:
            image_embedding = self.clip_encoder.encode_image(image_query)
            clip_results = self.clip_index.search(image_embedding, k=50)
            results.append(('clip', clip_results))
        
        # RRF fusion
        final_ranking = self.rrf_fusion(results)
        
        # Log search event to user profile
        self.user_profile_manager.log_search(
            user_id=user_id,
            query_text=text_query,
            query_image=image_query,
            results=final_ranking[:top_k]
        )
        
        return final_ranking[:top_k]
```

### Key Features
- ✅ Handles text-only, image-only, or multimodal queries
- ✅ Direct search (no behavioral bias initially)
- ✅ Results feed into user profile for future recommendations
- ✅ Answers teacher's question: "What if user uploads image?"

---

## Mode 2: Passive Recommendation (System-Initiated)

### Purpose
System proactively recommends items based on user's accumulated behavior profile.

### Pipeline

```
User Behavior Profile
    ↓
┌─────────────────────────────────────────┐
│  Step 1: Collaborative Filtering        │
│  • Option A: Cleora (current)           │
│  • Option B: RL-based CF (proposed)     │
│  • Generate 50-100 candidates           │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 2: Content Sanity Check           │
│  • BLaIR: Semantic relevance to profile │
│  • CLIP: Visual consistency             │
│  • Hard Veto: Remove low-scoring items  │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 3: RRF Re-ranking                 │
│  • Fuse BLaIR + CLIP scores             │
│  • Output: Personalized Top-K           │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Step 4: Track Engagement               │
│  • Record which items user clicked      │
│  • Update user profile continuously     │
└─────────────────────────────────────────┘
```

---

## User Behavior Profile Construction

### Teacher's Core Concern: "No connection between user interactions and system"

### Profile Components

```python
class UserBehaviorProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        
        # Interaction history
        self.searches = []          # Search queries (text + image)
        self.clicks = []            # Items clicked
        self.purchases = []         # Items purchased
        self.ratings = []           # Item ratings
        
        # Aggregated embeddings
        self.text_profile = None    # Averaged BLaIR embeddings
        self.visual_profile = None  # Averaged CLIP embeddings
        
        # Temporal features
        self.recent_interactions = deque(maxlen=50)
        self.long_term_preferences = {}
        
        # Behavioral patterns
        self.preferred_categories = Counter()
        self.activity_times = []
        self.diversity_score = 0.0
```

### Profile Update Logic

```python
class UserProfileManager:
    def log_search(self, user_id, query_text, query_image, results):
        """
        Record active search event
        """
        profile = self.get_profile(user_id)
        
        # Store search
        search_event = {
            'timestamp': datetime.now(),
            'query_text': query_text,
            'query_image': query_image,
            'results_shown': results,
            'modality': self._detect_modality(query_text, query_image)
        }
        profile.searches.append(search_event)
    
    def log_click(self, user_id, item_id, context):
        """
        Record click event (from search or recommendation)
        """
        profile = self.get_profile(user_id)
        
        click_event = {
            'timestamp': datetime.now(),
            'item_id': item_id,
            'source': context.get('source'),  # 'search' or 'recommendation'
            'position': context.get('position'),  # Rank in results
        }
        profile.clicks.append(click_event)
        
        # Update recent interactions
        profile.recent_interactions.append(item_id)
        
        # Trigger profile re-computation
        self.update_aggregated_embeddings(user_id)
    
    def update_aggregated_embeddings(self, user_id):
        """
        Re-compute user profile embeddings from interaction history
        """
        profile = self.get_profile(user_id)
        
        # Get all interacted items
        all_items = (
            [e['item_id'] for e in profile.clicks] +
            [e['item_id'] for e in profile.purchases]
        )
        
        if not all_items:
            return
        
        # Fetch item embeddings
        item_embeddings = self.db.get_embeddings(all_items)
        
        # Temporal weighting (recent items weighted higher)
        weights = self._compute_temporal_weights(profile.clicks)
        
        # Weighted average of BLaIR embeddings
        blair_embeddings = [emb['blair'] for emb in item_embeddings]
        profile.text_profile = np.average(blair_embeddings, axis=0, weights=weights)
        
        # Weighted average of CLIP embeddings
        clip_embeddings = [emb['clip_image'] for emb in item_embeddings]
        profile.visual_profile = np.average(clip_embeddings, axis=0, weights=weights)
        
        # Update category preferences
        categories = [self.db.get_category(item) for item in all_items]
        profile.preferred_categories = Counter(categories)
        
        # Save updated profile
        self.save_profile(profile)
    
    def _compute_temporal_weights(self, interactions):
        """
        Recent interactions weighted higher using exponential decay
        """
        now = datetime.now()
        weights = []
        
        for interaction in interactions:
            age_days = (now - interaction['timestamp']).days
            weight = np.exp(-0.1 * age_days)  # Decay rate: 0.1
            weights.append(weight)
        
        # Normalize
        weights = np.array(weights)
        return weights / weights.sum()
```

### Profile-Based Retrieval

```python
class ProfileBasedRetriever:
    def recommend_for_user(self, user_id, top_k=10):
        """
        Generate recommendations based on user behavior profile
        """
        profile = self.profile_manager.get_profile(user_id)
        
        if not profile.text_profile or not profile.visual_profile:
            # Cold start: return trending items
            return self.get_trending_items(top_k)
        
        # Step 1: Collaborative Filtering (Cleora or RL-CF)
        candidates = self.collaborative_filter(profile, top_n=100)
        
        # Step 2: Content Sanity Check
        verified_candidates = self.content_verify(
            candidates,
            user_text_profile=profile.text_profile,
            user_visual_profile=profile.visual_profile
        )
        
        # Step 3: RRF Fusion
        final_ranking = self.rrf_fusion(verified_candidates)
        
        return final_ranking[:top_k]
    
    def collaborative_filter(self, profile, top_n=100):
        """
        Option A: Cleora-based (current)
        Option B: RL-based CF (proposed alternative)
        """
        # Current: Multi-seed Cleora retrieval
        candidates = set()
        
        for item_id in profile.recent_interactions:
            neighbors = self.cleora_index.search(item_id, k=50)
            candidates.update(neighbors)
        
        # Remove items user already has
        candidates = candidates - set(profile.clicks)
        
        return list(candidates)[:top_n]
```

---

## Question: Can RL Replace Cleora?

### Your Question: "Can RL replace Cleora for better collaborative filtering?"

**Answer: Yes, but with trade-offs**

### Option A: Keep Cleora (Current)

**Pros**:
- ✅ Fast, deterministic
- ✅ No training needed
- ✅ Proven to work
- ✅ Simpler to explain

**Cons**:
- ❌ No learning from feedback
- ❌ Static embeddings
- ❌ Markov-based (no temporal dynamics)

### Option B: Replace with RL-Based Collaborative Filtering

**Approach**: Use Deep Q-Learning to learn user-item interaction patterns

**Architecture**:
```
User Profile State
    ↓
Deep Q-Network
    ↓
Q-values for candidate items
    ↓
Select Top-K items with highest Q-values
```

**Implementation**:

```python
class RLCollaborativeFilter:
    """
    Replaces Cleora with learned collaborative filtering
    """
    def __init__(self, item_catalog, state_dim=256, action_dim=1000):
        # State: User profile embedding
        # Action: Select item from catalog
        self.dqn = CollaborativeFilterDQN(state_dim, action_dim)
        self.item_catalog = item_catalog
    
    def get_candidates(self, user_profile, top_n=100):
        """
        Use DQN to select candidate items
        
        Args:
            user_profile: UserBehaviorProfile object
            top_n: Number of candidates to return
        
        Returns:
            List of item_ids
        """
        # Encode user state
        state = self._encode_user_state(user_profile)
        
        # Get Q-values for all items
        with torch.no_grad():
            q_values = self.dqn(state)
        
        # Select top-N items with highest Q-values
        top_indices = torch.topk(q_values, top_n).indices
        candidate_items = [self.item_catalog[i] for i in top_indices]
        
        return candidate_items
    
    def _encode_user_state(self, profile):
        """
        Convert user profile to state vector
        """
        state = np.concatenate([
            profile.text_profile,           # BLaIR embedding (128-dim)
            profile.visual_profile,         # CLIP embedding (128-dim)
            self._encode_preferences(profile),  # Category prefs, etc
        ])
        return torch.FloatTensor(state)
```

**Training**:

```python
class RLCFTrainer:
    def train(self, interaction_data, num_epochs=100):
        """
        Train RL-based collaborative filter
        
        Reward: +1 if recommended item was clicked/purchased
                 0 otherwise
        """
        for epoch in range(num_epochs):
            for user_id in interaction_data.user_ids:
                # Get user history
                history = interaction_data.get_history(user_id, until='80%')
                future = interaction_data.get_history(user_id, from='80%')
                
                # Build profile from history
                profile = self.build_profile(history)
                
                # Generate candidates
                candidates = self.rl_cf.get_candidates(profile, top_n=50)
                
                # Compute reward
                future_items = set(future['item_id'].values)
                rewards = [1 if item in future_items else 0 
                          for item in candidates]
                
                # Train DQN
                self.rl_cf.train_step(profile, candidates, rewards)
```

**Pros**:
- ✅ Learns from user feedback
- ✅ Adapts over time
- ✅ Can model temporal dynamics
- ✅ Directly answers RL requirement

**Cons**:
- ❌ Requires training
- ❌ More complex
- ❌ Need sufficient data
- ❌ Slower inference

---

## Recommendation: Hybrid Approach

### Keep Both, Use Adaptively

```python
class AdaptiveCollaborativeFilter:
    def __init__(self, cleora, rl_cf):
        self.cleora = cleora
        self.rl_cf = rl_cf
    
    def get_candidates(self, user_profile, top_n=100):
        """
        Use RL-CF for users with rich profiles
        Use Cleora for cold-start users
        """
        num_interactions = len(user_profile.clicks)
        
        if num_interactions >= 20:
            # Warm user: Use RL-CF
            candidates = self.rl_cf.get_candidates(user_profile, top_n)
        elif num_interactions >= 5:
            # Medium user: Blend both
            rl_candidates = self.rl_cf.get_candidates(user_profile, top_n // 2)
            cleora_candidates = self.cleora.get_candidates(user_profile, top_n // 2)
            candidates = list(set(rl_candidates + cleora_candidates))
        else:
            # Cold user: Use Cleora (more robust)
            candidates = self.cleora.get_candidates(user_profile, top_n)
        
        return candidates
```

---

## Complete System Flow

### Scenario 1: New User (Cold Start)

```
1. User arrives → No profile exists
2. Show trending items (popularity baseline)
3. User searches: "fantasy books" + [dragon cover image]
4. Active Search Mode activates:
   - BLaIR finds "fantasy books"
   - CLIP finds books with dragon covers
   - RRF merges results
5. User clicks on "Game of Thrones"
6. Profile created:
   - text_profile = BLaIR embedding of "Game of Thrones"
   - visual_profile = CLIP embedding of GoT cover
   - preferred_categories = ['fantasy']
7. Next visit → Recommendation Mode activates:
   - Cleora/RL-CF retrieves similar fantasy books
   - BLaIR/CLIP verify relevance to profile
   - User sees personalized recommendations
```

### Scenario 2: Returning User (Warm Start)

```
1. User has 50+ interactions in profile
2. Homepage loads → Recommendation Mode:
   - RL-CF (if implemented) generates candidates based on learned patterns
   - BLaIR/CLIP verify against user's aggregated profile
   - Display Top-10 personalized recommendations
3. User searches: [uploads book cover photo]
4. Active Search Mode:
   - CLIP image search finds visually similar books
   - Results logged to profile
5. Profile continuously updated with every interaction
```

---

## Updated Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                           │
│  Mode 1: Active Search (text/image query)                    │
│  Mode 2: Passive Browse (view recommendations)               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              USER BEHAVIOR PROFILE (Continuously Updated)     │
│  • Interaction history (clicks, searches, purchases)         │
│  • Aggregated embeddings (text_profile, visual_profile)      │
│  • Temporal weighting (recent = higher weight)               │
│  • Category preferences, diversity scores                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────────┐
│  MODE 1 SEARCH  │     │  MODE 2 RECOMMEND    │
└─────────────────┘     └──────────────────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌───────────────────────┐
│  BLaIR + CLIP    │    │  Cleora OR RL-CF      │
│  Direct Search   │    │  (Collaborative)      │
└─────────┬────────┘    └──────────┬────────────┘
          │                        │
          │                        ▼
          │             ┌──────────────────────┐
          │             │  BLaIR + CLIP        │
          │             │  Sanity Check        │
          │             └──────────┬───────────┘
          │                        │
          └────────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   RRF Fusion Layer    │
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   Final Rankings      │
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  User Interacts       │
           │  (Click, Purchase)    │
           └───────────┬───────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  Update User Profile    │
         │  (Feedback Loop)        │
         └─────────────────────────┘
```

---

## Key Innovations

### 1. Dual-Mode System
- ✅ Active search handles user queries (text/image)
- ✅ Passive recommendation uses behavioral profile
- ✅ Both modes feed same profile

### 2. Continuous Profile Learning
- ✅ Every interaction updates profile
- ✅ Temporal weighting (recent = more important)
- ✅ Addresses teacher's "disconnected" concern

### 3. Flexible Collaborative Filtering
- ✅ Can use Cleora (simple, fast)
- ✅ Can use RL-CF (learned, adaptive)
- ✅ Can blend both based on user maturity

### 4. Multimodal Query Support
- ✅ Text-only queries
- ✅ Image-only queries (teacher's question!)
- ✅ Hybrid text+image queries

---

## Implementation Priority

### Phase 1: Core System (4 weeks)
1. Active Search Mode (BLaIR + CLIP + RRF)
2. User Profile Manager
3. Cleora-based Recommendation Mode

### Phase 2: RL Enhancement (Optional, 3 weeks)
4. RL-based Collaborative Filter
5. Adaptive blending (Cleora + RL-CF)
6. Comparative evaluation

### Phase 3: Evaluation (2 weeks)
7. Offline metrics (temporal holdout)
8. Ablation studies
9. User profile quality analysis

---

## Answering Teacher's Concerns

**"No user connection"**
> ✅ User profile continuously built from all interactions (search + clicks + purchases)

**"Cleora isolated"**
> ✅ Cleora retrieves based on user profile, not random items

**"What if user uploads image?"**
> ✅ Active Search Mode handles image queries via CLIP

**"Markov → RL"**
> ✅ Can replace Cleora with RL-CF for learned collaborative filtering

---

**Project Status**: Revised Architecture  
**Focus**: Dual-mode (search + recommendation) with continuous profile learning  
**RL Role**: Optional replacement for Cleora in collaborative filtering layer  
**Last Updated**: March 2026
