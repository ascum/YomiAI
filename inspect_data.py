from datasets import load_from_disk

ds = load_from_disk('src/data/Books_meta_extracted/Books_meta')

total = 0
has_desc = 0
lengths = []
desc_samples = []

with open('inspect_output.txt', 'w', encoding='utf-8') as out:
    out.write(f"Dataset size: {len(ds)}\n")
    out.write(f"Columns: {ds.column_names}\n\n")

    for item in ds.select(range(3000)):
        total += 1
        desc = item.get('description', [])
        if desc and isinstance(desc, list) and any(v for v in desc if v and v.strip()):
            has_desc += 1
            text = ' '.join(str(v) for v in desc if v and v.strip())
            lengths.append(len(text))
            if len(desc_samples) < 5:
                title = (item.get('title') or '').strip().replace('\n', ' ').replace('\r', '')[:80]
                desc_clean = text.strip().replace('\n', ' ').replace('\r', '')[:250]
                desc_samples.append((title, desc_clean))

    out.write(f"=== DESCRIPTION FILL RATE (3000 sample) ===\n")
    out.write(f"Has description: {has_desc}/{total} = {has_desc/total*100:.1f}%\n")
    if lengths:
        out.write(f"Avg length: {sum(lengths)//len(lengths)} chars\n")
        out.write(f"Min: {min(lengths)} | Max: {max(lengths)}\n")
    out.write("\n=== SAMPLE DESCRIPTIONS ===\n")
    for title, desc in desc_samples:
        out.write(f"Title: {title}\n")
        out.write(f"Desc:  {desc}\n\n")

print("Done. See inspect_output.txt")
