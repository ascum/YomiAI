"""
app/repository/metadata_repo.py — Parquet metadata access layer.

Extracted from api.py (_get_item_details, metadata loading).
"""
import ast
import logging
import os
import pandas as pd

log = logging.getLogger("nba_api")


class MetadataRepository:
    """Loads item_metadata.parquet and provides per-ASIN detail lookups."""

    def __init__(self, data_dir: str):
        self._df: pd.DataFrame | None = None
        self._load(data_dir)

    def _load(self, data_dir: str):
        meta_path = os.path.join(data_dir, "item_metadata.parquet")
        try:
            df = pd.read_parquet(meta_path)
            if "parent_asin" in df.columns:
                df["parent_asin"] = df["parent_asin"].astype(str)
                df.set_index("parent_asin", inplace=True)
            else:
                df.index = df.index.map(str)
            self._df = df
            log.info(f"Metadata loaded: {len(df):,} items")
        except Exception as e:
            log.warning(f"Metadata not found — stub metadata active: {e}")
            self._df = pd.DataFrame()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def get_item(self, asin: str) -> dict:
        """Return a fully-hydrated item dict for the given ASIN."""
        if self._df is not None and asin in self._df.index:
            row = self._df.loc[asin]

            # Parse author name (may be a stringified dict from HuggingFace)
            author_val = row.get("author_name")
            clean_author = "Unknown Author"
            if author_val and str(author_val) != "nan":
                auth_str = str(author_val)
                if auth_str.startswith("{") and auth_str.endswith("}"):
                    try:
                        auth_dict = ast.literal_eval(auth_str)
                        clean_author = auth_dict.get("name", "Unknown Author")
                    except Exception:
                        clean_author = auth_str
                else:
                    clean_author = auth_str

            raw_desc = row.get("description", "")
            description = str(raw_desc).strip()[:300] if raw_desc and str(raw_desc) != "nan" else ""

            title_val = row.get("title")
            title = str(title_val) if title_val and str(title_val) != "nan" else f"Book {asin[:8]}"

            genre_val = row.get("main_category")
            genre = str(genre_val) if genre_val and str(genre_val) != "nan" else "Books"

            img_val = row.get("image_url")
            image_url = str(img_val) if img_val and str(img_val) != "nan" else None

            return {
                "id":          asin,
                "title":       title,
                "author":      clean_author,
                "genre":       genre,
                "image_url":   image_url,
                "description": description,
                "cover_color": "#" + hex(abs(hash(asin)) % 0xFFFFFF)[2:].zfill(6),
            }

        return {
            "id":          asin,
            "title":       f"Book {asin[:8]}",
            "author":      "Unknown Author",
            "genre":       "Books",
            "image_url":   None,
            "cover_color": "#" + hex(abs(hash(asin)) % 0xFFFFFF)[2:].zfill(6),
        }
