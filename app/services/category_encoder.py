"""
app/services/category_encoder.py — Category Vocabulary for DIF-SASRec

Parses the 'categories' field from item_metadata.parquet into a vocabulary
mapping. Uses LEAF categories (last pipe-separated segment) for sharpest
genre signal.

Category format in parquet: "Books|Literature & Fiction|Action & Adventure"
Leaf category extracted:    "Action & Adventure"
"""
import json

import pandas as pd


class CategoryEncoder:
    """
    Manages category vocabulary for the DIF-SASRec personal recommendation model.

    Special tokens:
        PAD_ID = 0  (padding for sequence alignment)
        UNK_ID = 1  (unknown / missing category)
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self):
        self.vocab: dict[str, int] = {}       # {category_string: int_id}
        self.id_to_cat: dict[int, str] = {}   # {int_id: category_string}
        self.asin_to_cat_id: dict[str, int] = {}  # {asin: int_id}
        self.num_categories: int = 2          # starts with PAD + UNK

    # ── Build from metadata ───────────────────────────────────────────────────

    def build_from_parquet(self, metadata_path: str):
        """
        Build category vocabulary from item_metadata.parquet.

        Reads 'parent_asin' and 'categories' columns.
        Extracts the leaf (last segment) of each pipe-separated category string.
        Assigns integer IDs starting from 2 (0=PAD, 1=UNK).
        """
        print(f"[CategoryEncoder] Loading metadata from {metadata_path} ...")
        df = pd.read_parquet(metadata_path, columns=["parent_asin", "categories"])

        cat_set: set[str] = set()
        asin_cats: dict[str, str] = {}

        for _, row in df.iterrows():
            asin = str(row["parent_asin"])
            raw  = str(row.get("categories", ""))
            leaf = self._parse_leaf_category(raw)
            if leaf:
                cat_set.add(leaf)
                asin_cats[asin] = leaf

        sorted_cats = sorted(cat_set)
        self.vocab = {cat: idx + 2 for idx, cat in enumerate(sorted_cats)}
        self.id_to_cat = {0: "PAD", 1: "UNK"}
        self.id_to_cat.update({idx + 2: cat for idx, cat in enumerate(sorted_cats)})
        self.num_categories = len(self.vocab) + 2  # +2 for PAD and UNK

        self.asin_to_cat_id = {
            asin: self.vocab.get(cat, self.UNK_ID)
            for asin, cat in asin_cats.items()
        }

        print(f"[CategoryEncoder] Vocabulary built: {self.num_categories} categories "
              f"({len(self.vocab)} unique + PAD + UNK)")
        print(f"[CategoryEncoder] ASINs with categories: {len(self.asin_to_cat_id):,}")

        from collections import Counter
        freq = Counter(asin_cats.values())
        print("[CategoryEncoder] Top 10 categories:")
        for cat, count in freq.most_common(10):
            print(f"    {cat}: {count:,}")

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_leaf_category(raw: str) -> str:
        """
        Extract the leaf category from a pipe-separated string.

        Examples:
            "Books|Literature & Fiction|Action & Adventure" → "Action & Adventure"
            "Books|Science Fiction" → "Science Fiction"
            "Books" → "Books"
            "" or "nan" → ""
        """
        if not raw or raw == "nan" or raw.strip() == "":
            return ""
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        return parts[-1] if parts else ""

    # ── Lookups ───────────────────────────────────────────────────────────────

    def get_category_id(self, asin: str) -> int:
        """Return the category ID for an ASIN. Returns UNK_ID if not found."""
        return self.asin_to_cat_id.get(asin, self.UNK_ID)

    def get_category_name(self, cat_id: int) -> str:
        """Reverse lookup: int_id → category string."""
        return self.id_to_cat.get(cat_id, "UNK")

    def encode_sequence(self, asin_sequence: list) -> list:
        """Convert a list of ASINs into a list of category IDs."""
        return [self.get_category_id(asin) for asin in asin_sequence]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save vocabulary to JSON."""
        payload = {
            "vocab":          self.vocab,
            "asin_to_cat_id": self.asin_to_cat_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CategoryEncoder] Saved to {path}")

    def load(self, path: str):
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.vocab          = payload["vocab"]
        self.asin_to_cat_id = payload.get("asin_to_cat_id", {})
        self.id_to_cat      = {0: "PAD", 1: "UNK"}
        self.id_to_cat.update({v: k for k, v in self.vocab.items()})
        self.num_categories = len(self.vocab) + 2
        print(f"[CategoryEncoder] Loaded {self.num_categories} categories from {path}")
