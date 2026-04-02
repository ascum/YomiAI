# Project Brief: Hybrid Discovery-Based Recommendation System

## Executive Summary

This project develops a production-ready recommendation system that combines **behavioral graph patterns** (Cleora) with **multimodal content analysis** (BLaIR + CLIP) using a three-layer "Candidate & Re-rank" pipeline. The system moves beyond traditional query-driven search to anticipate user needs while maintaining robustness against common recommendation flaws.

## Problem Statement

Modern recommendation systems face critical challenges:

1. **Single-Modality Limitations**:
   - Pure collaborative filtering (behavioral) can recommend irrelevant items due to noisy data or bot traffic
   - Pure content-based approaches miss valuable collaborative patterns from user behavior
   - Neither approach alone delivers optimal results

2. **Scalability vs Quality Trade-off**:
   - High-quality semantic matching is computationally expensive at scale
   - Fast retrieval often sacrifices relevance
   - Need for efficient multi-stage pipeline

3. **Robustness Issues**:
   - Popularity bias dominates recommendations
   - Context blindness leads to poor suggestions
   - Cold-start problems for new items/users

4. **Trust & Safety**:
   - Behavioral models can "hallucinate" completely unrelated recommendations
   - No verification mechanism to ensure recommendation quality

## Solution: Three-Layer Hybrid Pipeline

### Architecture Overview

Our system implements a **"Candidate & Re-rank"** pipeline with three distinct layers:

```
Query Item → [Layer 1: Cleora] → [Layer 2: BLaIR+CLIP] → [Layer 3: RRF] → Top-K Results
              (50-100 candidates)  (Content Verification)  (Fusion)
```

### Layer 1: Candidate Generation (The Broad Net)
**Engine**: Cleora graph embeddings  
**Role**: Rapid behavioral retrieval  
**Input**: User interaction graph (clicks, views, purchases)  
**Output**: Top 50-100 items most frequently associated with query item

**Why it works**:
- Extremely fast matrix multiplication (no neural network training needed)
- Captures "wisdom of the crowd" - hidden behavioral connections
- Example: "People who bought this camera also bought this specific SD card"

### Layer 2: Content Verification (The Sanity Check)
**Engines**: BLaIR (text) + CLIP (images)  
**Role**: Quality control and hallucination prevention  
**Input**: 50-100 candidates from Cleora  
**Output**: Filtered candidates with similarity scores

**Why it works**:
- Cleora is mathematically "blind" to content - only sees co-occurrence patterns
- BLaIR ensures semantic/textual relevance
- CLIP ensures visual similarity
- **Hard Veto Rule**: Candidates scoring <0.3 in BOTH models are discarded

### Layer 3: Hybrid Fusion (The Judge)
**Engine**: Reciprocal Rank Fusion (RRF)  
**Role**: Combine text and image rankings into unified list  
**Formula**: 
$$\text{Score} = \sum \frac{1}{k + \text{rank}}$$

**Why it works**:
- No complex tuning required (parameter-free)
- Pushes consensus items (high in both modalities) to top
- Preserves items strong in just one modality
- Inherently balances different signal types

## Technical Implementation

### Data Engineering: The K-Core Filter

**Problem**: Sparse graphs produce noisy embeddings  
**Solution**: Enforce dense subgraph via K-Core filtering

**Algorithm**:
1. Remove users with <5 interactions
2. Remove items with <5 interactions  
3. Iterate until no more nodes removable

**Result**: Dense, high-quality interaction graph with no isolated "islands"

### Interaction Weighting Strategy

Different interaction types carry different weights:
- **View**: 1 repetition in edge list
- **Add to Cart**: 3 repetitions
- **Purchase**: 5 repetitions

This emphasizes high-value interactions without complex weighting schemes.

### Edge Case Handling

**Cold Start - New Items** (no Cleora embedding):
- Bypass Layer 1 entirely
- Query directly using BLaIR + CLIP content embeddings
- Fallback to pure content-based recommendation

**Cold Start - New Users** (no interaction history):
- Display global trending items (SQL-based counts)
- Alternative: Diverse category sampling
- Once user clicks → activate full pipeline

**Hard Veto Triggers** (all candidates fail threshold):
- Expand candidate pool to top-100
- Temporarily relax threshold to 0.2
- Log cases for analysis and improvement

## Dataset & Scale

### Target Dataset Requirements
- **Source**: Amazon Reviews 2023 or equivalent behavioral dataset
- **Post-Filtering Scale**:
  - 100,000+ interactions
  - 1,000-10,000 users
  - 500-5,000 items

### Multimodal Embeddings
- **Source**: Hugging Face Hub (`minhkhang26/my-nba-project-data`)
- **Format**: 90 NPZ chunks
- **Embeddings**:
  - BLaIR: 1024-dim text embeddings
  - CLIP Text: 512-dim text embeddings
  - CLIP Image: 512-dim visual embeddings

## Evaluation Strategy

### Ablation Studies (Core)

**Ablation 1**: RRF vs Pure Cleora
- **Question**: Does content verification improve quality?
- **Metrics**: Precision@K, NDCG@K, Diversity
- **Hypothesis**: RRF prevents hallucinated recommendations

**Ablation 2**: RRF vs Pure BLaIR/CLIP
- **Question**: Does behavioral signal improve relevance?
- **Metrics**: Precision@K, Recall@K, MRR
- **Hypothesis**: Cleora captures collaborative patterns content alone misses

**Ablation 3**: Hard Veto Threshold Sensitivity
- **Test**: Thresholds 0.2, 0.3, 0.4
- **Analysis**: Precision vs Recall trade-off
- **Goal**: Find optimal balance

**Ablation 4**: Candidate Pool Size
- **Test**: Top-30, Top-50, Top-100 from Cleora
- **Analysis**: Quality vs latency trade-off
- **Goal**: Optimal candidate pool size

### Evaluation Metrics

**Relevance Metrics**:
- Precision@K: % relevant recommendations
- Recall@K: % of user's items covered
- NDCG@K: Ranking quality with position weighting
- MRR: First relevant item position

**Diversity Metrics**:
- Category Coverage: % unique categories
- Intra-List Diversity: Avg pairwise distance
- Catalog Coverage: % items ever recommended

**Business Metrics**:
- Popularity Bias: Distribution analysis
- Cold-Start Performance: Quality for new items/users

## Expected Outcomes

### Quantitative Targets
- **Precision@10**: >15% improvement over pure Cleora
- **NDCG@10**: >20% improvement over pure content
- **Diversity**: >25% increase in category coverage
- **Latency**: <200ms end-to-end
- **Coverage**: >80% catalog reachable

### Deliverables

1. **Production System**
   - FastAPI backend with <200ms latency
   - React frontend with interactive recommendations
   - Redis caching for hot items
   - Comprehensive API documentation

2. **Evaluation Results**
   - Complete ablation study report
   - Metric comparisons vs baselines
   - Visualization of improvements
   - Edge case analysis

3. **Documentation**
   - Complete thesis documenting:
     - Data engineering pipeline
     - Three-layer architecture
     - Implementation details
     - Evaluation methodology
     - Results and insights
   - Architecture diagrams
   - Embedding visualizations

4. **Defense Materials**
   - 20-minute presentation
   - Demo video (5-10 minutes)
   - Backup slides for Q&A

## Innovation & Contributions

### Novel Aspects

1. **Three-Layer Safety Architecture**: Unique combination of fast behavioral retrieval with content-based verification
2. **Hard Veto Mechanism**: Prevents hallucinated recommendations while maintaining recall
3. **Parameter-Free Fusion**: RRF requires no tuning while achieving strong performance
4. **Edge Case Robustness**: Comprehensive fallback strategies for real-world scenarios

### Research Contributions

1. **Empirical Validation**: Rigorous ablation studies proving hybrid superiority
2. **Scalability Analysis**: Demonstrate production-ready latency at scale
3. **Design Patterns**: Reusable patterns for hybrid recommender systems
4. **Open Questions**: Cold-start strategies, fusion alternatives, online learning

## Implementation Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| **Data Engineering** | 2 weeks | K-Core filtered graph with 100K+ interactions |
| **Cleora Layer** | 2 weeks | Behavioral embeddings + vector DB |
| **Content Layer** | 2 weeks | BLaIR/CLIP scoring engines |
| **Fusion & Pipeline** | 2 weeks | End-to-end RRF pipeline with edge cases |
| **Evaluation** | 2 weeks | Complete ablation studies |
| **Productization** | 2 weeks | FastAPI backend + React frontend |
| **Documentation** | 3 weeks | Thesis + demo video |
| **Defense Prep** | 1 week | Presentation + rehearsal |

**Total Duration**: ~16 weeks

## Technical Stack

**Core Engine**:
- Cleora (graph embeddings)
- BLaIR (text embeddings)
- CLIP (multimodal embeddings)

**Infrastructure**:
- Vector DB: FAISS or Annoy
- Backend: FastAPI (Python)
- Frontend: React
- Caching: Redis
- Database: PostgreSQL
- Analytics: DuckDB

**Evaluation**:
- scikit-learn (metrics)
- matplotlib/plotly (visualization)
- t-SNE/UMAP (embedding viz)

## Risk Mitigation

### Technical Risks

**Risk**: Cleora embeddings poor quality due to sparse data
- **Mitigation**: K-Core filtering ensures density
- **Fallback**: Increase minimum interaction threshold

**Risk**: Content verification too strict (low recall)
- **Mitigation**: Tune hard veto threshold via ablation
- **Fallback**: Adaptive thresholds based on candidate quality

**Risk**: RRF doesn't outperform baselines
- **Mitigation**: Try alternative fusion (weighted average, learned)
- **Evidence**: RRF proven effective in literature

### Project Risks

**Risk**: Dataset insufficient scale
- **Mitigation**: Amazon Reviews 2023 has millions of interactions
- **Fallback**: MovieLens, Goodreads, or synthetic augmentation

**Risk**: Latency exceeds target
- **Mitigation**: Caching, precomputation, batch processing
- **Evidence**: FAISS achieves <10ms for million-scale vectors

## Future Work

### Short-Term Extensions
- **Learned Fusion**: Replace RRF with neural fusion model
- **Contextual Features**: Incorporate time, location, device
- **Multi-Objective**: Balance accuracy, diversity, novelty

### Long-Term Research
- **Online Learning**: Real-time embedding updates
- **Bandits**: Exploration-exploitation with LinUCB
- **Cross-Domain**: Transfer learning across product categories
- **Explainability**: Why this recommendation? (visual + behavioral)

## Team Responsibilities

- **Member A**: Data engineering, K-Core filtering, graph construction
- **Member B**: Cleora implementation, behavioral layer, vector DB setup
- **Member C**: Content layer (BLaIR/CLIP), RRF fusion, edge cases
- **All Members**: Evaluation, ablations, documentation, frontend

## Key References

- **Cleora**: Fast, scalable graph embedding framework
- **BLaIR**: Text embeddings for semantic understanding
- **CLIP**: Multimodal vision-language model
- **RRF**: Reciprocal Rank Fusion algorithm
- **Amazon Reviews 2023**: Large-scale behavioral dataset

---

**Project Status**: Phase 2 Implementation  
**Architecture**: Three-Layer Candidate & Re-rank Pipeline  
**Last Updated**: February 2026  
**Contact**: nguyeenkhanhs.cs@gmail.com, Danko Pham