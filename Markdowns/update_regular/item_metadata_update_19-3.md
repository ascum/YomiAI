# Item Metadata & BGE-Reranker Update (March 19)

This document details the recent structural updates to the core `item_metadata.parquet` database utilized by the multimodal recommendation system, specifically tailored to support the new **BAAI/bge-reranker-v2-m3** cross-encoder integration.

---

## 📊 1. What Changed Compared to the Old Metadata?

Historically, the system operated on a tightly constrained version of the dataset. The previous `item_metadata.parquet` file lacked rich textual descriptions because the dual-encoder retrieval phase (BLaIR + FAISS) relied solely on pre-computed vector indexes, making the storage of massive text blocks unnecessary.

With the introduction of the **BGE-Reranker** (a true Cross-Encoder), the system now must re-read the *actual book text* at runtime alongside the user's query to evaluate deep semantic interactions via self-attention mechanisms. Thus, the database had to be radically expanded.

### Key Upgrades:
1. **Sample Size Doubled:** The extraction limit from the raw `Books_meta` Arrow dataset was increased from `1,000,000` to `2,000,000` items to maximize catalog coverage.
2. **The `description` Column:** A completely new data dimension was piped into the parquet file. The extraction pipeline (`build_metadata_cache.py`) was rewritten to parse nested `numpy.ndarray` string lists from the Arrow format, join them into human-readable blurbs, and carefully cap them at 1,000 characters to prevent Memory/VRAM blowouts during cross-encoding.

---

## 📈 2. Current Parquet Database Statistics

Following the successful rebuild workflow, the new core engine database (`item_metadata.parquet`) possesses the following metrics:

- **Total Unique Rows Iterated & Consolidated:** `1,732,910` items
- **Total Columns Per Row:** `6` (up from 5)
- **Rows with Populated Descriptions:** `820,684` items
- **Description Fill Rate:** `47.36%` (matches the inherent density of the raw Amazon Books dataset)

---

## 🗂️ 3. What Does Each Row Contain?

Every item in the system's runtime memory (`api.py` -> `_state["metadata_df"]`) now cleanly loads the following schema:

| Column Name | Data Type | Description | Example Data |
| :--- | :--- | :--- | :--- |
| `parent_asin` | `string` | The unique Amazon Standard Identification Number serving as the primary index key. | `"1588464040"` |
| `title` | `string` | The full name of the book. | `"Dark Ages Mage"` |
| `author_name` | `string` | Extracted author string (defaults to "Unknown Author" if missing). | `"Unknown Author"` |
| `main_category` | `string` | The primary genre taxonomy. | `"Books"` |
| `image_url` | `string` | A direct link to the high-resolution Amazon book cover (used by the React frontend). | `"https://images.../xyz.jpg"` |
| `description` **[NEW]** | `string` | The author's blurb, plot summary, or editorial review. Capped to the first 1,000 characters. | `"A guide to appropriate instruction, workshops that focus... [truncated]"` |

---

## ⚙️ 4. How the Integration Works in Production

The expanded metadata enables the following pipeline execution upon every user query in Mode 1 (Active Search):

1. **Retrieval**: BLaIR and CLIP pull the top 50 semantic and top 50 visual nearest neighbors, fusing them via Reciprocal Rank Fusion (RRF).
2. **Data Hydration**: The system takes the top 20 candidates from the RRF output and looks up their `title` and newly added `description` fields from the 1.73M-row `metadata_df` loaded in RAM.
3. **Cross-Encoding Execution**: 
   - A string containing both fields is bundled with the user's raw text query: `[[query, "Title: X. Description: Y"]]`.
   - The `BGEReranker` (via `sentence-transformers.CrossEncoder`) mathematically processes this combined input.
4. **Final Delivery**: Books are re-sorted by their new `reranker_score`. The `api.py` endpoint yields the exact precision of the cross-encoder, while additionally transmitting up to 300 characters of the new description to the React frontend UI to enrich the user experience.

> **Note:** For items lacking a description (the remaining ~52%), the reranker gracefully falls back to evaluating the query purely against the book's `title`.
