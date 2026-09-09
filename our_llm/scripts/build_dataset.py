from pathlib import Path
import re
import hashlib
import json
from collections import Counter

PROJECT_ROOT = Path("../")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

for directory in [RAW_DIR, CLEANED_DIR, FILTERED_DIR, FINAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


raw_file = RAW_DIR / "fitness_train.jsonl"
cleaned_file = CLEANED_DIR / "fitness_cleaned.jsonl"
filtered_file = FILTERED_DIR / "fitness_filtered.jsonl"

raw_docs = load_jsonl(raw_file)
cleaned_docs = load_jsonl(cleaned_file)
filtered_docs = load_jsonl(filtered_file)

raw_chars = sum(len(doc["text"]) for doc in raw_docs)

cleaning_stats = {
    "input": len(raw_docs),
    "output": len(cleaned_docs),
    "empty_removed": len(raw_docs) - len(cleaned_docs)
}

FILTER_CONFIG = {
    "min_chars": 200,
    "max_chars": 100000,
    "min_alpha_ratio": 0.30,
    "max_repeated_line_ratio": 0.30
}


def alphabetic_ratio(text):
    if not text:
        return 0

    alpha = sum(char.isalpha() for char in text)

    return alpha / len(text)


def repeated_line_ratio(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0

    counts = Counter(lines)
    repeated = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return repeated / len(lines)


filter_stats = {
    "input": len(cleaned_docs),
    "min_chars": 0,
    "max_chars": 0,
    "alpha_ratio": 0,
    "repeated_lines": 0,
    "output": len(filtered_docs)
}

for doc in cleaned_docs:
    text = doc["text"]

    if len(text) < FILTER_CONFIG["min_chars"]:
        filter_stats["min_chars"] += 1
        continue

    if len(text) > FILTER_CONFIG["max_chars"]:
        filter_stats["max_chars"] += 1
        continue

    if alphabetic_ratio(text) < FILTER_CONFIG["min_alpha_ratio"]:
        filter_stats["alpha_ratio"] += 1
        continue

    if repeated_line_ratio(text) > FILTER_CONFIG["max_repeated_line_ratio"]:
        filter_stats["repeated_lines"] += 1
        continue


def document_hash(text):
    normalized = text.strip().lower()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


unique_docs = []
seen_hashes = set()
duplicate_count = 0

for doc in filtered_docs:
    text = doc["text"]
    doc_hash = document_hash(text)

    if doc_hash in seen_hashes:
        duplicate_count += 1
        continue

    seen_hashes.add(doc_hash)
    unique_docs.append(doc)

print("Input documents:", len(filtered_docs))
print("Duplicates removed:", duplicate_count)
print("Unique documents:", len(unique_docs))

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"
)

NUMBER_PATTERN = re.compile(
    r"(?<!\d)\d{6,}(?!\d)"
)


def scrub_pii(text):
    stats = {
        "email": 0,
        "phone": 0,
        "digits": 0
    }

    text, stats["email"] = EMAIL_PATTERN.subn(
        "<EMAIL>",
        text
    )

    text, stats["phone"] = PHONE_PATTERN.subn(
        "<PHONE>",
        text
    )

    text, stats["digits"] = NUMBER_PATTERN.subn(
        "<NUMBER>",
        text
    )

    return text, stats


final_docs = []

pii_stats = {
    "email": 0,
    "phone": 0,
    "digits": 0
}

for doc in unique_docs:
    cleaned_text, stats = scrub_pii(doc["text"])

    for key in pii_stats:
        pii_stats[key] += stats[key]

    final_docs.append({
        "text": cleaned_text
    })

print("PII matches:")
print(pii_stats)

final_file = FINAL_DIR / "fitness_final.jsonl"

with open(final_file, "w", encoding="utf-8") as f:
    for doc in final_docs:
        f.write(
            json.dumps(doc, ensure_ascii=False) + "\n"
        )

print(f"Final corpus saved to: {final_file}")

final_chars = sum(
    len(doc["text"])
    for doc in final_docs
)

stats_report = f"""# Data Pipeline Statistics

## Raw

- Documents: {len(raw_docs)}
- Characters: {raw_chars}
- Size: {raw_file.stat().st_size / (1024**2):.2f} MB

## Cleaning

- Input documents: {cleaning_stats["input"]}
- Output documents: {cleaning_stats["output"]}
- Empty/broken documents removed: {cleaning_stats["empty_removed"]}

## Filtering

- Input documents: {filter_stats["input"]}
- Removed by min_chars: {filter_stats["min_chars"]}
- Removed by max_chars: {filter_stats["max_chars"]}
- Removed by alphabetic ratio: {filter_stats["alpha_ratio"]}
- Removed by repeated-line ratio: {filter_stats["repeated_lines"]}
- Output documents: {filter_stats["output"]}

## Deduplication

- Input documents: {len(filtered_docs)}
- Exact duplicates removed: {duplicate_count}
- Unique documents: {len(unique_docs)}

## PII Scrubbing

- Emails masked: {pii_stats["email"]}
- Phone numbers masked: {pii_stats["phone"]}
- Large digit sequences masked: {pii_stats["digits"]}

## Final

- Documents: {len(final_docs)}
- Characters: {final_chars}

## Pipeline

Raw
→ Cleaning
→ Filtering
→ Deduplication
→ PII Scrubbing
→ Final Corpus
"""

report_file = FINAL_DIR / "stats_report.md"

with open(report_file, "w", encoding="utf-8") as f:
    f.write(stats_report)

print(stats_report)
print(f"Report saved to: {report_file}")

stage_counts = {
    "Raw": len(raw_docs),
    "Cleaned": len(cleaned_docs),
    "Filtered": len(filtered_docs),
    "Deduplicated": len(unique_docs),
    "Final": len(final_docs)
}

for stage, count in stage_counts.items():
    retention = count / len(raw_docs) * 100

    print(
        f"{stage:15} : "
        f"{count:,} documents "
        f"({retention:.2f}% retained)"
    )

for i in range(min(5, len(raw_docs))):
    print("=" * 80)
    print("RAW:")
    print(raw_docs[i]["text"][:500])

    print("\nCLEANED:")
    print(cleaned_docs[i]["text"][:500])

for i, doc in enumerate(final_docs[:5]):
    print("=" * 80)
    print(f"FINAL DOCUMENT {i + 1}")
    print(doc["text"][:1000])