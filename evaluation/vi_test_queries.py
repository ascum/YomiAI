"""
evaluation/vi_test_queries.py
Ground truth for Vietnamese queries and English equivalents.
"""

VI_TEST_QUERIES = [
    {
        "query": "tiểu thuyết trinh thám",
        "expected_genres": ["detective", "mystery", "crime", "thriller", "suspense"]
    },
    {
        "query": "sách phát triển bản thân",
        "expected_genres": ["self-help", "habits", "mindset", "success", "motivational"]
    },
    {
        "query": "fantasy phép thuật",
        "expected_genres": ["magic", "wizard", "fantasy", "sword", "sorcery"]
    },
    {
        "query": "lịch sử thế chiến",
        "expected_genres": ["world war", "history", "1939", "military", "wwii", "war"]
    },
    {
        "query": "sách thiếu nhi",
        "expected_genres": ["children", "kids", "picture book", "fairy tale"]
    }
]

EN_EQUIVALENT_QUERIES = [
    "detective mystery novels",
    "self-help personal development books",
    "magic fantasy",
    "world war history",
    "children's books"
]
