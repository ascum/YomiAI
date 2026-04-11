# Phase 2

## Hybrid Discovery-Based Recommendation System

> This phase focuses on building a production-ready recommendation system using a three-layer "Candidate & Re-rank" pipeline. The system combines behavioral graph patterns (Cleora) with multimodal content analysis (BLaIR + CLIP) to deliver personalized, contextually relevant recommendations.

### System Architecture Overview

**Pipeline Design**: Three-layer candidate generation and re-ranking system

1. **Layer 1 - Candidate Generation (Behavioral Base)**: Cleora graph embeddings for fast retrieval
2. **Layer 2 - Content Verification (Sanity Check)**: BLaIR + CLIP for semantic/visual filtering
3. **Layer 3 - Hybrid Fusion (The Judge)**: Reciprocal Rank Fusion (RRF) for final ranking

### Key Design Principles

1. **Discovery-Based**: Anticipate user needs rather than pure query-driven search
2. **Scalable**: Fast candidate generation with efficient re-ranking
3. **Robust**: Resistant to popularity bias, bot traffic, and context blindness
4. **Safe**: Content verification layer prevents hallucinated recommendations

### Focus On

**Core deliverable**: Working hybrid recommendation system using Cleora (behavioral) + BLaIR/CLIP (content) with RRF fusion, demonstrating superiority over single-modality baselines through ablation studies.

## Data Infrastructure

### Multimodal Embeddings Dataset

The project uses pre-computed multimodal embeddings stored in Hugging Face Hub for content-based verification.

**Dataset Structure:**
- Repository: `minhkhang26/my-nba-project-data`
- Format: NPZ chunks (90 chunks total, indexed 0-89)
- Location: `meta_img_merged_chunks/meta_img_merged_chunk_XXX.npz`
- Fields per chunk:
  - **asin**: Amazon Standard Identification Number (product ID)
  - **emb_blair**: BLaIR embeddings (text-based semantic features)
  - **emb_clip_text**: CLIP text embeddings
  - **emb_clip_img**: CLIP image embeddings

**Loading Example:**
```python
import os
from huggingface_hub import login, hf_hub_download
import numpy as np
import pandas as pd

# Authenticate (Recommended: Use .env file or environment variables)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)   

# Download specific chunk
repo_id = "minhkhang26/my-nba-project-data"
filename = "meta_img_merged_chunks/meta_img_merged_chunk_000.npz"

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset"
)

# Load and structure data
data = np.load(local_path, allow_pickle=True)
df = pd.DataFrame({
    "asin": data["asin"],
    "emb_blair": list(data["emb_blair"]),
    "emb_clip_text": list(data["emb_clip_text"]),
    "emb_clip_img": list(data["emb_clip_img"]),
})

print(f"Loaded chunk with {df.shape[0]} items")
```

**Batch Loading Strategy:**
```python
def load_all_chunks(repo_id, num_chunks=90):
    """Load all embedding chunks from Hugging Face"""
    all_data = []
    
    for i in range(num_chunks):
        filename = f"meta_img_merged_chunks/meta_img_merged_chunk_{i:03d}.npz"
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
        data = np.load(local_path, allow_pickle=True)
        chunk_df = pd.DataFrame({
            "asin": data["asin"],
            "emb_blair": list(data["emb_blair"]),
            "emb_clip_text": list(data["emb_clip_text"]),
            "emb_clip_img": list(data["emb_clip_img"]),
        })
        all_data.append(chunk_df)
        print(f"Loaded chunk {i+1}/{num_chunks}")
    
    return pd.concat(all_data, ignore_index=True)
```

## Milestones

- Milestone 1: Data Engineering & Graph Construction

  **Goal:** Prepare behavioral interaction data for Cleora
  
  **Description:**
  
  > Build a dense bipartite graph from user interaction logs, applying rigorous filtering to ensure high-quality embeddings.
  
  #### Epic 1.1: Dataset Acquisition & Initial Exploration
  
  - [ ] Acquire Amazon Reviews 2023 dataset (or equivalent behavioral dataset)
  - [ ] Explore data structure: users, items, interaction types, timestamps
  - [ ] Identify interaction types: views, clicks, add-to-cart, purchases, ratings
  - [ ] Document data statistics: total users, items, interactions
  - [ ] **Deliverable**: Raw dataset loaded with initial statistics report
  
  #### Epic 1.2: K-Core Filtering (The Dense Subgraph Rule)
  
  - [ ] Implement K-Core filtering algorithm (K=5 minimum interactions)
  - [ ] Apply filter to both users AND items:
    - [ ] Remove users with < 5 interactions
    - [ ] Remove items with < 5 interactions
    - [ ] Iterate until no more nodes can be removed
  - [ ] Validate post-filtering dataset meets minimum scale:
    - [ ] 100,000+ interactions
    - [ ] 1,000-10,000 users
    - [ ] 500-5,000 items
  - [ ] Document impact: % data retained, connectivity improvements
  - [ ] **Deliverable**: Filtered dense subgraph dataset
  
  #### Epic 1.3: Weighted Edge List Construction
  
  - [ ] Design weighting scheme based on interaction type:
    - [ ] View = 1 repetition
    - [ ] Add to Cart = 3 repetitions
    - [ ] Purchase = 5 repetitions
  - [ ] Generate TSV edge list format for Cleora:
    ```
    User_ID    Product_ID
    user_123   item_456
    user_123   item_456    # Purchase (repeated 5x)
    user_123   item_456
    user_123   item_456
    user_123   item_456
    ```
  - [ ] Validate edge list: check for duplicates, verify format
  - [ ] Store edge list in efficient format (TSV/CSV)
  - [ ] **Deliverable**: Weighted edge list ready for Cleora input

- Milestone 2: Layer 1 - Cleora Behavioral Embeddings

  **Goal:** Generate behavioral embeddings capturing "wisdom of the crowd"
  
  **Description:**
  
  > Run Cleora on the weighted interaction graph to produce fast, behavior-based item embeddings.
  
  #### Epic 2.1: Cleora Configuration & Execution
  
  - [ ] Install Cleora and dependencies
  - [ ] Configure Cleora parameters:
    - [ ] Embedding dimension: 1024 (balance between quality and speed)
    - [ ] Number of iterations: tune based on convergence
    - [ ] Complex types: define entity types (User, Item)
  - [ ] Run Cleora on weighted edge list
  - [ ] Monitor training: convergence, training time, memory usage
  - [ ] **Deliverable**: Cleora embeddings for all items and users
  
  #### Epic 2.2: Behavioral Embedding Validation
  
  - [ ] Quality checks:
    - [ ] Verify embedding dimensions and coverage
    - [ ] Check for NaN or infinite values
    - [ ] Validate all filtered items have embeddings
  - [ ] Semantic validation:
    - [ ] Test behavioral similarity: do co-purchased items cluster?
    - [ ] Visualize embeddings (t-SNE/UMAP) - check for meaningful clusters
    - [ ] Spot-check: query known item, verify neighbors make sense
  - [ ] **Deliverable**: Validated Cleora embeddings with quality report
  
  #### Epic 2.3: Vector Database Setup
  
  - [ ] Choose vector DB solution (FAISS, Annoy, or dedicated solution)
  - [ ] Index Cleora embeddings for fast nearest neighbor search
  - [ ] Benchmark query performance (latency, throughput)
  - [ ] Set retrieval parameters: top-K candidates (50-100)
  - [ ] **Deliverable**: Behavioral vector DB with sub-100ms query latency

- Milestone 3: Layer 2 - Multimodal Content Embeddings

  **Goal:** Prepare content-based verification layer
  
  **Description:**
  
  > Load and index BLaIR and CLIP embeddings for semantic and visual similarity scoring.
  
  #### Epic 3.1: Content Embedding Integration
  
  - [ ] Load all 90 NPZ chunks from Hugging Face (see Data Infrastructure section)
  - [ ] Merge embeddings into unified dataset
  - [ ] Align with Cleora data: ensure ASIN coverage matches
  - [ ] Handle missing embeddings: document coverage gaps
  - [ ] **Deliverable**: Complete multimodal embedding dataset
  
  #### Epic 3.2: BLaIR Text Similarity Engine
  
  - [ ] Index BLaIR embeddings in vector DB (separate index from Cleora)
  - [ ] Implement similarity scoring function:
    ```python
    def blair_score(query_item, candidate_items):
        # Cosine similarity between query and candidates
        return cosine_similarity(query_emb, candidate_embs)
    ```
  - [ ] Set similarity threshold: 0.3 minimum (hard veto)
  - [ ] Test on sample queries: verify text semantic matching works
  - [ ] **Deliverable**: BLaIR scoring function with threshold validation
  
  #### Epic 3.3: CLIP Visual Similarity Engine
  
  - [ ] Index CLIP image embeddings in vector DB
  - [ ] Implement visual similarity scoring:
    ```python
    def clip_score(query_item, candidate_items):
        # Cosine similarity using CLIP image embeddings
        return cosine_similarity(query_img_emb, candidate_img_embs)
    ```
  - [ ] Set similarity threshold: 0.3 minimum (hard veto)
  - [ ] Test visual similarity: verify visually similar items rank high
  - [ ] **Deliverable**: CLIP scoring function with threshold validation

- Milestone 4: Layer 3 - Hybrid Fusion & Pipeline Integration

  **Goal:** Combine all three layers into end-to-end pipeline
  
  **Description:**
  
  > Implement Reciprocal Rank Fusion (RRF) to merge behavioral and content signals, with safety mechanisms.
  
  #### Epic 4.1: Reciprocal Rank Fusion (RRF) Implementation
  
  - [ ] Implement RRF algorithm:
    ```python
    def reciprocal_rank_fusion(rankings_list, k=60):
        """
        rankings_list: List of ranked item lists from different models
        k: Constant for RRF formula (default 60)
        Returns: Fused ranking
        """
        scores = {}
        for ranking in rankings_list:
            for rank, item_id in enumerate(ranking):
                if item_id not in scores:
                    scores[item_id] = 0
                scores[item_id] += 1 / (k + rank + 1)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ```
  - [ ] Test RRF with sample rankings: verify consensus items rank higher
  - [ ] Tune k parameter (try k=50, 60, 70) based on empirical results
  - [ ] **Deliverable**: Working RRF fusion function
  
  #### Epic 4.2: End-to-End Pipeline Construction
  
  - [ ] Build complete retrieval pipeline:
    ```python
    def recommend(query_item_id, top_k=5):
        # Layer 1: Cleora candidate generation
        candidates = cleora_retrieve(query_item_id, top_n=50)
        
        # Layer 2: Content verification (Hard Veto)
        blair_scores = blair_score(query_item_id, candidates)
        clip_scores = clip_score(query_item_id, candidates)
        
        # Safety valve: remove items scoring < 0.3 in BOTH models
        valid_candidates = [
            c for c in candidates 
            if blair_scores[c] >= 0.3 or clip_scores[c] >= 0.3
        ]
        
        # Get ranked lists from content models
        blair_ranked = rank_by_score(valid_candidates, blair_scores)
        clip_ranked = rank_by_score(valid_candidates, clip_scores)
        
        # Layer 3: RRF fusion
        final_ranking = reciprocal_rank_fusion([blair_ranked, clip_ranked])
        
        return final_ranking[:top_k]
    ```
  - [ ] Test complete pipeline with diverse queries
  - [ ] Monitor performance: end-to-end latency target < 200ms
  - [ ] **Deliverable**: Working end-to-end recommendation pipeline
  
  #### Epic 4.3: Edge Case Handling
  
  - [ ] **Cold Start - New Items** (no Cleora embedding):
    - [ ] Implement fallback: skip Layer 1, use only BLaIR + CLIP
    - [ ] Direct similarity search in content embedding space
    - [ ] Test with newly added items
  - [ ] **Cold Start - New Users** (no interaction history):
    - [ ] Implement fallback: global trending items (SQL query on interaction counts)
    - [ ] Alternative: diverse category sampling
    - [ ] Once user clicks first item → activate full pipeline
  - [ ] **Low Quality Candidates** (all fail hard veto):
    - [ ] Implement secondary fallback: expand candidate pool to top-100
    - [ ] Relax threshold temporarily to 0.2
    - [ ] Log these cases for later analysis
  - [ ] **Deliverable**: Robust pipeline with all edge cases handled

- Milestone 5: Evaluation & Ablation Studies

  **Goal:** Prove hybrid model superiority
  
  **Description:**
  
  > Conduct rigorous evaluation comparing hybrid pipeline against single-modality baselines.
  
  #### Epic 5.1: Test Set Preparation
  
  - [ ] Create held-out test set:
    - [ ] Temporal split: last 20% of interactions by timestamp
    - [ ] Ensure test users have sufficient history in training set
  - [ ] Define test queries: recent user interactions to predict
  - [ ] Document test set statistics
  - [ ] **Deliverable**: Clean train/test split with validation
  
  #### Epic 5.2: Baseline Implementations
  
  - [ ] **Baseline 1**: Pure Cleora (no content filtering)
    - [ ] Return top-K from behavioral embeddings only
  - [ ] **Baseline 2**: Pure BLaIR + CLIP (no behavioral signal)
    - [ ] Average BLaIR and CLIP similarities, no Cleora
  - [ ] **Baseline 3**: Popularity-based
    - [ ] Recommend most-viewed/purchased items globally
  - [ ] **Baseline 4**: Random
    - [ ] Random sampling from catalog
  - [ ] **Deliverable**: Four baseline systems implemented
  
  #### Epic 5.3: Evaluation Metrics Implementation
  
  - [ ] Implement standard metrics:
    - [ ] **Precision@K**: % of recommendations that user interacted with
    - [ ] **Recall@K**: % of user's interactions that were recommended
    - [ ] **NDCG@K**: Normalized Discounted Cumulative Gain
    - [ ] **MRR**: Mean Reciprocal Rank (where first relevant item appears)
  - [ ] Implement diversity metrics:
    - [ ] **Category Coverage**: % of unique categories in recommendations
    - [ ] **Intra-List Diversity**: Average pairwise distance in recommendation list
  - [ ] Implement business metrics:
    - [ ] **Catalog Coverage**: % of items ever recommended
    - [ ] **Popularity Bias**: Distribution of item popularity in recommendations
  - [ ] **Deliverable**: Complete evaluation framework
  
  #### Epic 5.4: Ablation Studies Execution
  
  - [ ] **Ablation 1**: RRF vs Pure Cleora
    - [ ] Compare: Does content verification improve quality?
    - [ ] Measure: Precision, NDCG, Diversity
    - [ ] Hypothesis: RRF prevents hallucinated recommendations
  
  - [ ] **Ablation 2**: RRF vs Pure BLaIR/CLIP
    - [ ] Compare: Does behavioral signal improve relevance?
    - [ ] Measure: Precision, Recall, MRR
    - [ ] Hypothesis: Cleora captures collaborative patterns content alone misses
  
  - [ ] **Ablation 3**: Hard Veto Threshold Sensitivity
    - [ ] Test thresholds: 0.2, 0.3, 0.4
    - [ ] Compare: Precision vs Recall trade-off
    - [ ] Find optimal threshold balancing quality and coverage
  
  - [ ] **Ablation 4**: Candidate Pool Size
    - [ ] Test: Top-30, Top-50, Top-100 from Cleora
    - [ ] Compare: Quality vs latency trade-off
    - [ ] Determine optimal candidate pool size
  
  - [ ] **Deliverable**: Comprehensive ablation study report with visualizations

- Milestone 6: System Productization

  **Goal:** Deploy production-ready system
  
  **Description:**
  
  > Build FastAPI backend and React frontend for live demonstration.
  
  #### Epic 6.1: Backend API Development
  
  - [ ] Design FastAPI endpoints:
    - [ ] `POST /recommend`: Get recommendations for item
    - [ ] `GET /item/{item_id}`: Fetch item details
    - [ ] `GET /trending`: Get global trending items
    - [ ] `POST /feedback`: Log user interactions
  - [ ] Implement caching layer (Redis) for hot items
  - [ ] Add monitoring: latency tracking, error rates
  - [ ] Write API documentation (OpenAPI/Swagger)
  - [ ] **Deliverable**: Production FastAPI backend
  
  #### Epic 6.2: Frontend Development
  
  - [ ] Design UI components:
    - [ ] Item detail page with "Similar Items" section
    - [ ] Recommendation explanations (behavioral vs content)
    - [ ] Visual indicators for recommendation source
  - [ ] Implement React frontend:
    - [ ] Item browsing interface
    - [ ] Real-time recommendation display
    - [ ] Interaction tracking (clicks, views)
  - [ ] Connect frontend to backend API
  - [ ] **Deliverable**: Interactive React demo application
  
  #### Epic 6.3: Performance Optimization
  
  - [ ] Profile system bottlenecks:
    - [ ] Vector DB query times
    - [ ] Embedding loading times
    - [ ] RRF computation overhead
  - [ ] Optimize hot paths:
    - [ ] Batch processing for multiple queries
    - [ ] Embedding caching strategies
    - [ ] Precompute popular item neighborhoods
  - [ ] Load testing: ensure system handles 100+ concurrent queries
  - [ ] **Deliverable**: Optimized system meeting <200ms latency target
  
  #### Epic 6.4: Visualization & Debugging Tools
  
  - [ ] Build internal tools:
    - [ ] Embedding space visualization (t-SNE/UMAP)
    - [ ] Recommendation debugging UI (show layer-by-layer results)
    - [ ] Similarity score inspector
  - [ ] Create demonstration materials:
    - [ ] Edge case showcase (cold start, hard veto examples)
    - [ ] Ablation study visualizations
    - [ ] System architecture diagrams
  - [ ] **Deliverable**: Comprehensive debugging and demo toolkit

- Milestone 7: Documentation & Defense Preparation

  **Goal:** Complete thesis and prepare defense
  
  **Description:**
  
  > Document all work, results, and insights for thesis defense.
  
  #### Epic 7.1: Thesis Writing
  
  - [ ] **Chapter 1**: Introduction
    - [ ] Problem statement: limitations of single-modality recommenders
    - [ ] Proposed solution: three-layer hybrid pipeline
    - [ ] Contributions summary
  
  - [ ] **Chapter 2**: Related Work
    - [ ] Graph-based recommendation (collaborative filtering)
    - [ ] Content-based recommendation (embeddings)
    - [ ] Hybrid approaches (fusion strategies)
  
  - [ ] **Chapter 3**: System Architecture
    - [ ] **Member A**: Data Engineering & Graph Construction
    - [ ] **Member B**: Cleora Behavioral Layer
    - [ ] **Member C**: Content Verification Layer & RRF
  
  - [ ] **Chapter 4**: Implementation Details
    - [ ] Pipeline implementation
    - [ ] Edge case handling
    - [ ] Performance optimization
  
  - [ ] **Chapter 5**: Evaluation & Results
    - [ ] Ablation study results
    - [ ] Metric comparisons
    - [ ] Error analysis and limitations
  
  - [ ] **Chapter 6**: Conclusion & Future Work
    - [ ] Key findings
    - [ ] Limitations and lessons learned
    - [ ] Future directions (online learning, neural fusion)
  
  - [ ] **Deliverable**: Complete thesis draft
  
  #### Epic 7.2: Demo Video & Materials
  
  - [ ] Record demo video (5-10 minutes):
    - [ ] System walkthrough
    - [ ] Example queries with explanations
    - [ ] Edge case demonstrations
    - [ ] Ablation study visualizations
  - [ ] Prepare supplementary materials:
    - [ ] Architecture diagrams
    - [ ] Embedding visualizations
    - [ ] Performance benchmarks
  - [ ] **Deliverable**: Professional demo video and materials
  
  #### Epic 7.3: Defense Presentation
  
  - [ ] Prepare slide deck (20-25 slides):
    - [ ] Problem & Motivation (3 slides)
    - [ ] System Architecture (5 slides)
    - [ ] Three-Layer Pipeline Deep Dive (6 slides)
    - [ ] Evaluation Results (4 slides)
    - [ ] Ablation Studies (3 slides)
    - [ ] Limitations & Future Work (2 slides)
  - [ ] Rehearse presentation:
    - [ ] 20-minute main talk
    - [ ] 10-minute Q&A practice
  - [ ] Prepare backup slides for anticipated questions:
    - [ ] Why RRF instead of learned fusion?
    - [ ] How does K-Core filtering affect fairness?
    - [ ] Computational cost analysis
  - [ ] **Deliverable**: Polished defense presentation

## Technical Stack

- **Data Processing**: Python (pandas, numpy), DuckDB for fast analytics
- **Graph Embeddings**: Cleora
- **Vector Storage**: FAISS or Annoy (for fast ANN search)
- **Content Embeddings**: Pre-computed (BLaIR, CLIP) from Hugging Face
- **Backend**: FastAPI (Python)
- **Caching**: Redis
- **Frontend**: React
- **Database**: PostgreSQL (user logs, item metadata)
- **Visualization**: matplotlib, plotly, t-SNE/UMAP

## Success Criteria

### Quantitative Metrics
- **Precision@10**: >15% improvement over pure Cleora baseline
- **NDCG@10**: >20% improvement over pure content baseline
- **Diversity**: >25% increase in category coverage vs popularity baseline
- **Latency**: <200ms end-to-end query latency
- **Coverage**: >80% of catalog reachable via recommendations

### Qualitative Validation
- Visual inspection: recommendations "make sense" to human evaluators
- Edge cases handled gracefully (cold start, hard veto scenarios)
- System stable under load testing

## Metadata
- URL: [https://linear.app/dacn/project/phase-2-51724b0ab70e](https://linear.app/dacn/project/phase-2-51724b0ab70e)
- Status: In Progress
- Lead: No lead
- Members: nguyeenkhanhs.cs@gmail.com, Danko Pham
- Start date: Not set
- Target date: Not set