from pathlib import Path
import re
import hashlib
import json
import unicodedata
import html
from collections import Counter

from datasets import load_dataset


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path("../")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

for directory in [RAW_DIR, CLEANED_DIR, FILTERED_DIR, FINAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT.resolve())
print("Raw:", RAW_DIR.resolve())
print("Cleaned:", CLEANED_DIR.resolve())
print("Filtered:", FILTERED_DIR.resolve())
print("Final:", FINAL_DIR.resolve())


# ============================================================
# JSONL utilities
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def save_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )


# ============================================================
# 1. Download dataset and create RAW corpus
# ============================================================

from datasets import load_dataset

dataset = load_dataset(
    "chibbss/fitness-chat-prompt-completion-dataset",
    split="train",
)

print(dataset)
print(dataset.column_names)

if len(dataset) > 0:
    print(dataset[0])

if len(dataset) > 2:
    print(dataset[2])

raw_file = RAW_DIR / "fitness_train.jsonl"

with open(raw_file, "w", encoding="utf-8") as f:
    for item in dataset:
        record = {
            "text": f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print(f"Saved raw data to: {raw_file}")
print(f"File size: {raw_file.stat().st_size / (1024**2):.2f} MB")


# ============================================================
# 2. Load RAW corpus and collect statistics
# ============================================================

raw_docs = load_jsonl(raw_file)

raw_chars = sum(
    len(doc["text"])
    for doc in raw_docs
)

print("Number of raw documents:", len(raw_docs))
print("Raw characters:", raw_chars)
print("Raw size MB:", round(raw_file.stat().st_size / (1024**2), 2))


# ============================================================
# 3. Cleaning
# ============================================================

def remove_html(text):
    """
    Remove HTML/XML-like tags.
    """
    return re.sub(r"<[^>]+>", " ", text)


def normalize_unicode(text):
    """
    Normalize Unicode using NFKC.
    """
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text):
    """
    Remove control characters while keeping newline and tab.
    """
    return "".join(
        char
        for char in text
        if char in "\n\t"
        or not unicodedata.category(char).startswith("C")
    )


def normalize_whitespace(text):
    """
    Compress repeated whitespace.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def clean_text(text):
    """
    Complete cleaning pipeline.
    """
    text = html.unescape(text)
    text = remove_html(text)
    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)

    return text


cleaned_docs = []

cleaning_stats = {
    "input": 0,
    "output": 0,
    "empty_removed": 0
}

for doc in raw_docs:
    cleaning_stats["input"] += 1

    original_text = doc["text"]
    cleaned_text = clean_text(original_text)

    if not cleaned_text:
        cleaning_stats["empty_removed"] += 1
        continue

    cleaned_docs.append({
        "text": cleaned_text
    })

cleaning_stats["output"] = len(cleaned_docs)

print(cleaning_stats)

cleaned_file = CLEANED_DIR / "fitness_cleaned.jsonl"
save_jsonl(cleaned_file, cleaned_docs)

print(f"Saved cleaned data to: {cleaned_file}")


# ============================================================
# 4. Filtering
# ============================================================

FILTER_CONFIG = {
    "min_chars": 200,
    "max_chars": 100000,
    "min_alpha_ratio": 0.30,
    "max_repeated_line_ratio": 0.30
}


def alphabetic_ratio(text):
    """
    Percentage of characters that are alphabetic.
    """
    if not text:
        return 0.0

    alpha_count = sum(
        char.isalpha()
        for char in text
    )

    return alpha_count / len(text)


def repeated_line_ratio(text):
    """
    Ratio of repeated non-empty lines.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) <= 1:
        return 0.0

    counts = Counter(lines)

    repeated_lines = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return repeated_lines / len(lines)


filtered_docs = []

filter_stats = {
    "input": len(cleaned_docs),
    "min_chars": 0,
    "max_chars": 0,
    "alpha_ratio": 0,
    "repeated_lines": 0,
    "output": 0
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

    filtered_docs.append(doc)

filter_stats["output"] = len(filtered_docs)

print(filter_stats)

filtered_file = FILTERED_DIR / "fitness_filtered.jsonl"
save_jsonl(filtered_file, filtered_docs)

print(f"Saved filtered data to: {filtered_file}")


# ============================================================
# 5. Deduplication
# ============================================================

def document_hash(text):
    """
    Create a SHA-256 hash from normalized document text.
    """
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


# ============================================================
# 6. PII scrubbing
# ============================================================

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
    """
    Replace emails, phone numbers, and long digit sequences.
    """
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
    final_text, stats = scrub_pii(doc["text"])

    for key in pii_stats:
        pii_stats[key] += stats[key]

    final_docs.append({
        "text": final_text
    })

print("PII matches:")
print(pii_stats)


# ============================================================
# 7. Save FINAL corpus
# ============================================================

final_file = FINAL_DIR / "fitness_final.jsonl"
save_jsonl(final_file, final_docs)

print(f"Final corpus saved to: {final_file}")

final_chars = sum(
    len(doc["text"])
    for doc in final_docs
)


# ============================================================
# 8. Statistics report
# ============================================================

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


# ============================================================
# 9. Retention statistics
# ============================================================

stage_counts = {
    "Raw": len(raw_docs),
    "Cleaned": len(cleaned_docs),
    "Filtered": len(filtered_docs),
    "Deduplicated": len(unique_docs),
    "Final": len(final_docs)
}

for stage, count in stage_counts.items():
    retention = (
        count / len(raw_docs) * 100
        if raw_docs
        else 0
    )

    print(
        f"{stage:15} : "
        f"{count:,} documents "
        f"({retention:.2f}% retained)"
    )


# ============================================================
# 10. Preview data at different stages
# ============================================================

for i in range(min(5, len(raw_docs))):
    print("=" * 80)
    print("RAW:")
    print(raw_docs[i]["text"][:100])

    print("\nCLEANED:")
    print(cleaned_docs[i]["text"][:100])

for i, doc in enumerate(final_docs[:5]):
    print("=" * 80)
    print(f"FINAL DOCUMENT {i + 1}")
    print(doc["text"][:1000])
