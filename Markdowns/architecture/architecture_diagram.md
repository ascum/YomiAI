# NBA Multimodal Recommendation System Pipeline

This document contains Mermaid diagrams illustrating the execution pipeline of your DATN project. You can share these with your teacher to explain the system architecture.

## 1. High-Level Architecture & User Actions

This diagram shows how the frontend interacts with the FastAPI backend across the three main flows: Active Search, Passive Recommendations, and Profile Interactions.

```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#3b82f6,stroke:#1e3a8a,stroke-width:2px,color:#fff;
    classDef api fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef engine fill:#8b5cf6,stroke:#5b21b6,stroke-width:2px,color:#fff;
    classDef db fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;

    UI[React Frontend]:::frontend

    subgraph FastAPI Backend
        API_S[POST /search]:::api
        API_R[GET /recommend]:::api
        API_I[POST /interact]:::api
    end

    UI -->|1. Text/Image Query| API_S
    UI -->|2. Request Feed| API_R
    UI -->|3. Click / Skip| API_I

    API_S --> ActiveSearchEngine:::engine
    API_R --> PassiveRecommendationEngine:::engine
    API_I --> UserProfileManager:::engine

    UserProfileManager --> RL[RL-DQN Agent]:::engine
    
    subgraph Data Layer
        FAISS[(FAISS Indices\nBLaIR & CLIP)]:::db
        CLEORA[(Cleora Graph\nEmbeddings)]:::db
        META[(Item Metadata)]:::db
        PROF[(User Profiles\nJSON)]:::db
    end

    ActiveSearchEngine -.-> FAISS
    PassiveRecommendationEngine -.-> CLEORA
    PassiveRecommendationEngine -.-> FAISS
    UserProfileManager -.-> PROF
```

---



## 2. Passive Recommendation Timeline (The 3-Layer NBA Funnel)

This is the most complex part of your system—the 3 layers of AI filtering that generate the personalized feed.

```mermaid
flowchart TD
    classDef start_end fill:#374151,stroke:#111827,stroke-width:2px,color:#fff;
    classDef layer1 fill:#8b5cf6,stroke:#5b21b6,stroke-width:2px,color:#fff;
    classDef layer2 fill:#ec4899,stroke:#9d174d,stroke-width:2px,color:#fff;
    classDef layer3 fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef data fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;

    Start([User Requests Feed]):::start_end
    LoadProf[Load User Profile\nClicks & Skips]:::data
    Start --> LoadProf
    
    ColdStart{Clicks < Threshold?}:::data
    LoadProf --> ColdStart

    ColdStart -->|Yes| Random[Serve Top Popular/Discovery Items]:::start_end
    ColdStart -->|No| Layer1

    subgraph Layer 1: Behavioral Generation
        Layer1[Get 5 Recent Clicks]:::layer1 --> CleoraSearch
        CleoraSearch[Lookup nearest neighbors in\nCleora Hypergraph Index]:::layer1
    end

    subgraph Layer 2: Content Veto
        CleoraSearch --> SanityCheck[Compare Candidates against\nUser's Aggregated Profiles]:::layer2
        SanityCheck --> SimCheck{Similarity > Threshold?}:::layer2
        SimCheck -->|No| Reject[Discard Item]
        SimCheck -->|Yes| Keep[Pass to RRF]:::layer2
    end

    subgraph Layer 3: RL & Final Fusion
        LoadRL[Load User-Specific DQN Weights]:::layer3
        Keep --> LoadRL
        LoadRL --> ScoreRL[DQN evaluates Reward Probability\nState=User Profile, Action=Item]:::layer3
        ScoreRL --> RRF[Reciprocal Rank Fusion\nCombine BLaIR + CLIP + DQN Scores]:::layer3
        RRF --> Tagging[Tag Item with Highest Contributing Model]:::layer3
    end

    Tagging --> End([Return Personalized Top-5 Feed]):::start_end
```

---

## 3. Real-time Reinforcement Learning (RL) Loop

How the system learns from user behavior in real-time without restarting.

```mermaid
sequenceDiagram
    participant UI as React Frontend
    participant API as /interact Endpoint
    participant Profile as UserProfileManager
    participant RL as RL-DQN Agent
    participant Disk as Local Storage (JSON/.pt)

    UI->>API: User clicks a Book (Reward = 1.0)
    API->>Profile: log_click(user_id, item_id)
    Profile->>Disk: Auto-save Profile State to JSON
    Profile-->>API: (Profile Updated)
    
    API->>RL: train_rl(user_profile, item_id, reward=1.0)
    Note over RL: Calculate Loss & Backpropagate
    RL-->>API: (Weights Updated in GPU Memory)
    
    API->>Disk: save_rl_weights(user_id_dqn.pt)
    API-->>UI: 200 OK (Status logged, RL updated)
```
