from pathlib import Path
import re
import hashlib
import json
import unicodedata
import html
from collections import Counter

import pandas as pd

# Project root
PROJECT_ROOT = Path("../")

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

# Create directories if they don't exist
for directory in [RAW_DIR, CLEANED_DIR, FILTERED_DIR, FINAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT.resolve())
print("Raw:", RAW_DIR.resolve())
print("Cleaned:", CLEANED_DIR.resolve())
print("Filtered:", FILTERED_DIR.resolve())
print("Final:", FINAL_DIR.resolve())

from datasets import load_dataset

dataset = load_dataset(
    "chibbss/fitness-chat-prompt-completion-dataset",
    split="train",
    
)

print(dataset)
print(dataset.column_names)
print(dataset[0])
print(dataset.column_names)
print(dataset[2])
len(dataset)
raw_file = RAW_DIR / "fitness_train.jsonl"

with open(raw_file, "w", encoding="utf-8") as f:
    for item in dataset:
        record = {
            "text": f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved raw data to: {raw_file}")
print(f"File size: {raw_file.stat().st_size / (1024**2):.2f} MB")
#Row data statistics
def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


raw_docs = load_jsonl(raw_file)

print("Number of raw documents:", len(raw_docs))
raw_chars = sum(len(doc["text"]) for doc in raw_docs)

print("Raw documents:", len(raw_docs))
print("Raw characters:", raw_chars)
print("Raw size MB:", round(raw_file.stat().st_size / (1024**2), 2))

def remove_html(text):
    """
    Remove HTML/XML-like tags.
    """
    text = re.sub(r"<[^>]+>", " ", text)    
    return text


def normalize_unicode(text):
    """
    Normalize Unicode using NFKC.
    """
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text):
    """
    Remove control characters while keeping
    newline and tab.
    """
    return "".join(
        char for char in text
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
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
#Statitisktika uchun
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
#tozalangan datani saqlash
cleaned_file = CLEANED_DIR / "fitness_cleaned.jsonl"

with open(cleaned_file, "w", encoding="utf-8") as f:
    for doc in cleaned_docs:
        f.write(
            json.dumps(doc, ensure_ascii=False) + "\n"
        )

print(f"Saved cleaned data to: {cleaned_file}")

FILTER_CONFIG = {
    "min_chars": 200,
    "max_chars": 100000,
    "min_alpha_ratio": 0.30,
    "max_repeated_line_ratio": 0.30
}

FILTER_CONFIG
def alphabetic_ratio(text):
    """
    Percentage of characters that are alphabetic.
    """
    if not text:
        return 0.0

    alpha_count = sum(char.isalpha() for char in text)

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
#Statistika uchun
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

    # Minimum length
    if len(text) < FILTER_CONFIG["min_chars"]:
        filter_stats["min_chars"] += 1
        continue

    # Maximum length
    if len(text) > FILTER_CONFIG["max_chars"]:
        filter_stats["max_chars"] += 1
        continue

    # Alphabetic ratio
    if alphabetic_ratio(text) < FILTER_CONFIG["min_alpha_ratio"]:
        filter_stats["alpha_ratio"] += 1
        continue

    # Repeated line ratio
    if repeated_line_ratio(text) > FILTER_CONFIG["max_repeated_line_ratio"]:
        filter_stats["repeated_lines"] += 1
        continue

    filtered_docs.append(doc)

filter_stats["output"] = len(filtered_docs)

print(filter_stats)
#fiter bolgan datani saqlash
filtered_file = FILTERED_DIR / "fitness_filtered.jsonl"

with open(filtered_file, "w", encoding="utf-8") as f:
    for doc in filtered_docs:
        f.write(
            json.dumps(doc, ensure_ascii=False) + "\n"
        )

print(f"Saved filtered data to: {filtered_file}")