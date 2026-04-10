# NBA Multimodal Recommendation System: Architecture Guide

This document outlines the system architecture following the FastAPI-native refactor (v3.0). It is designed to be modular, testable, and easily extensible.

---

## 🏗️ System Overview

The system follows a **FastAPI-native Repository-Service-Route** pattern, utilizing **Dependency Injection (DI)** to manage long-lived resources (ML models, FAISS indices, database connections).

### 📁 Directory Structure

```text
app/
├── api/            # HTTP Layer (FastAPI routes & Pydantic schemas)
├── core/           # System Foundation (DI Container, Lifespan, ML model loaders)
├── infrastructure/ # External IO (Database, Redis, Translation services)
├── repository/     # Data Access (FAISS, Metadata, Profiles)
└── services/       # Business Logic (Search, Recommendation engines, RL Filter)
```

---

## 🛠️ Key Architectural Components

### 1. The `AppContainer` (`app/core/container.py`)
A typed dataclass that acts as a central registry for all runtime dependencies. It replaces global variables and provides a single point of truth for the system's state.

### 2. Lifespan Management (`app/core/lifespan.py`)
Handles the system's startup and shutdown. It initializes the `AppContainer`, loads heavy ML models (CLIP, BLaIR) into memory/VRAM, and starts background workers (e.g., the Redis-to-Mongo logging worker).

### 3. Dependency Injection (`app/api/dependencies.py`)
Routes request the `AppContainer` via FastAPI's `Depends(require_ready)`. This ensures:
- **Loose Coupling:** Routes don't know *how* a model is loaded, only *that* it exists in the container.
- **Safety:** The `require_ready` dependency raises a `503 Service Unavailable` if the system is still initializing.

---

## 🔄 How to Extend or Interchange Components

### Scenario A: Adding a New Service
1. **Define the Logic:** Create a new class in `app/services/new_service.py`.
2. **Register in Container:** Add a field to the `AppContainer` class in `app/core/container.py`.
3. **Initialize in Lifespan:** In `app/core/lifespan.py`, instantiate your service and attach it to the container.
4. **Use in Route:** Inject the container into your route and call `container.new_service.do_something()`.

### Scenario B: Swapping an ML Model (e.g., Replacing CLIP)
1. **Update Loader:** Modify `app/core/models.py` to include a loader for the new model.
2. **Update Lifespan:** Change the model initialization in `app/core/lifespan.py`.
3. **Update Route/Service Logic:** If the new model has a different output shape or encoding method, update the corresponding logic in `app/api/routes/search.py` or the relevant service in `app/services/`.

### Scenario C: Replacing the Search Engine (e.g., FAISS to Qdrant)
1. **Create New Repository:** Implement a new repository in `app/repository/qdrant_repo.py` that matches the interface expected by your services.
2. **Update Service:** Modify `app/services/active_search.py` to accept the new repository.
3. **Update Container & Lifespan:** Update the `AppContainer` type hints and the `lifespan` initialization logic to use the new repository.

---

## 🚀 Development Best Practices

- **Keep Routes Lean:** Routes should handle request validation, orchestration, and response formatting. Complex logic belongs in **Services**.
- **Use `anyio.to_thread` for CPU-bound tasks:** When calling heavy ML model inference (which is synchronous in many libraries), wrap the call in `await anyio.to_thread.run_sync(...)` to avoid blocking the FastAPI event loop.
- **Background Tasks:** Use the `lifespan` background worker for tasks that don't need to return a value to the user (e.g., logging, async indexing).
- **Schemas:** Always define request and response models in `app/api/schemas.py` to ensure consistent API documentation (Swagger/OpenAPI).
