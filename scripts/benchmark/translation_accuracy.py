import asyncio
import logging
import sys
import time
from typing import List, Tuple

# Ensure app is in path
sys.path.append(".")

from app.infrastructure.translation import translate_to_en, detect_language

# Configure logging to be less noisy during the test
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("translation_test")

# Test cases: (Source Text, Expected Keywords/Meaning, Language Name)
TEST_CASES = [
    # Vietnamese
    ("tiểu thuyết trinh thám", ["detective", "novel"], "Vietnamese"),
    ("sách nấu ăn ngon", ["cook", "book"], "Vietnamese"),
    ("lịch sử thế giới", ["history", "world"], "Vietnamese"),
    
    # French
    ("livres de cuisine française", ["french", "cook", "book"], "French"),
    ("roman policier", ["detective", "novel", "police"], "French"),
    
    # German
    ("backbuch für kinder", ["baking", "children", "book"], "German"),
    ("geschichte der wissenschaft", ["history", "science"], "German"),
    
    # Spanish
    ("novela de ciencia ficción", ["science", "fiction", "novel"], "Spanish"),
    ("libros de autoayuda", ["self-help", "books"], "Spanish"),
    
    # Chinese (Simplified)
    ("科幻小说", ["science", "fiction", "novel"], "Chinese"),
    ("中餐食谱", ["chinese", "food", "recipe"], "Chinese"),
    
    # Japanese
    ("ミステリー小説", ["mystery", "novel"], "Japanese"),
    ("日本の歴史", ["japanese", "history"], "Japanese"),
    
    # Korean
    ("공포 소설", ["horror", "novel"], "Korean"),
    ("한국 요리 책", ["korean", "cooking", "book"], "Korean"),
]

async def run_benchmark():
    print(f"{'='*80}")
    print(f"{'Language':<15} | {'Source Text':<30} | {'Translation'}")
    print(f"{'-'*15} | {'-'*30} | {'-'*31}")
    
    passed = 0
    total = len(TEST_CASES)
    
    # Warmup
    translate_to_en("hello")
    
    start_time = time.perf_counter()
    
    for source, keywords, lang_name in TEST_CASES:
        t_start = time.perf_counter()
        
        # 1. Detect language
        detected_iso = detect_language(source)
        
        # 2. Translate
        translation = translate_to_en(source)
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        
        # 3. Simple Accuracy Check (Keyword matching)
        # Note: NLLB might use synonyms, so this is a "soft" check
        found_keywords = [k for k in keywords if k.lower() in translation.lower()]
        is_accurate = len(found_keywords) > 0
        
        if is_accurate:
            passed += 1
            status = "✓"
        else:
            status = "✗"
            
        print(f"{lang_name:<15} | {source:<30} | {translation}")
        if not is_accurate:
             print(f"{' ' * 18} [Expected one of: {keywords}]")
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print(f"{'='*80}")
    print(f"RESULTS:")
    print(f"Total Tests: {total}")
    print(f"Passed (Soft): {passed}")
    print(f"Accuracy Rate: {(passed/total)*100:.1f}%")
    print(f"Average Latency: {(total_time/total)*1000:.1f}ms per query")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
