"""
TrainerAI_LLM - multi-source data preprocessing.

Pipeline:

    A1/A2/A3/B1/B2/B3/FineWeb
        ↓
    source normalization
        ↓
    cleaning
        ↓
    filtering
        ↓
    exact deduplication
        ↓
    PII scrubbing
        ↓
    fitness synthetic programs
        ↓
    60% FineWeb + 40% Fitness mixture
        ↓
    final/mixed_final.jsonl

Important:
- run_01.yaml (model architecture) is not changed here.
- vocab_size remains 8000.
- Mixture weights are controlled by configs/data.yaml.
- This stage uses character count as a practical pre-tokenizer approximation.
- The tokenizer stage should report the final token-level distribution.
"""

from pathlib import Path
import hashlib
import html
import json
import random
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict

from datasets import load_dataset


# ============================================================
# 0. Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
SOURCE_DIR = RAW_DIR / "sources"

for directory in [RAW_DIR, CLEANED_DIR, FILTERED_DIR, FINAL_DIR, SOURCE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

random.seed(42)


# ============================================================
# 1. Config
# ============================================================

CONFIG = {
    "fineweb": {
        "dataset": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "split": "train",
        "max_documents": 5000,
    },

    "fitness_sources": {
        "A1": {
            "repo": "https://github.com/hasaneyldrm/exercises-dataset.git",
            "files": ["data/exercises.json"],
        },

        "A2": {
            "repo": "https://github.com/ExerciseDB/exercisedb-api.git",

            # The repository itself is API source/documentation, not a
            # checked-in 11k-exercise JSON dump. If no local JSON exists,
            # the script uses this V1 API endpoint.
            "api_url": (
                "https://exercisedb-api-navy.vercel.app/"
                "api/v1/exercises"
            ),
        },

        "A3": {
            "repo": "https://github.com/longhaul-fitness/exercises.git",
            "files": [
                "strength.json",
                "cardio.json",
                "flexibility.json",
            ],
        },
    },

    "hf_sources": {
        "B1": (
            "chibbss/fitness-chat-prompt-completion-dataset",
            "train",
        ),
        "B2": (
            "RepDB/exercise-dataset",
            "train",
        ),
        "B3": (
            "hammamwahab/fitness-qa",
            "train",
        ),
    },

    "mixture": {
        "general": 0.60,
        "fitness": 0.40,

        "exercise_knowledge": 0.20,
        "workout_programs": 0.40,
        "synthetic_programs": 0.30,
        "fitness_qa": 0.10,
    },

    "filters": {
        # FineWeb is normal web text.
        "fineweb_min_chars": 200,

        # Structured fitness records can be shorter.
        "fitness_min_chars": 80,

        "max_chars": 100000,
        "min_alpha_ratio": 0.30,
        "max_repeated_line_ratio": 0.30,
    },

    "synthetic": {
        # Deliberately large enough to fill the requested 30% category
        # without manually writing hundreds/thousands of programs.
        "target_documents": 5000,
    },
}


# ============================================================
# 2. Utilities
# ============================================================

def save_jsonl(path, docs):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )


def clean_text(text):
    text = html.unescape(str(text))

    # Remove HTML/XML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Unicode normalization.
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters but keep newline/tab.
    text = "".join(
        ch
        for ch in text
        if ch in "\n\t"
        or not unicodedata.category(ch).startswith("C")
    )

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def as_text(value):
    """
    Convert structured values into text.

    Important:
    - strings stay strings
    - lists are joined
    - dicts prefer English
    - dict/list values are NOT silently discarded
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple)):
        parts = []

        for item in value:
            item_text = as_text(item)

            if item_text:
                parts.append(item_text)

        return "\n".join(parts)

    if isinstance(value, dict):

        # For multilingual instruction dictionaries use English first.
        if isinstance(value.get("en"), str):
            english = value["en"].strip()

            if english:
                return english

        parts = []

        for key, item in value.items():
            item_text = as_text(item)

            if item_text:
                parts.append(f"{key}: {item_text}")

        return "\n".join(parts)

    return str(value).strip()


def join_values(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
            str(x).strip()
            for x in value
            if str(x).strip()
        )

    return str(value).strip()


def alphabetic_ratio(text):
    if not text:
        return 0.0

    return sum(
        ch.isalpha()
        for ch in text
    ) / len(text)


def repeated_line_ratio(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) <= 1:
        return 0.0

    counts = Counter(lines)

    repeated = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return repeated / len(lines)


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
        "digits": 0,
    }

    text, stats["email"] = EMAIL_PATTERN.subn(
        "<EMAIL>",
        text,
    )

    text, stats["phone"] = PHONE_PATTERN.subn(
        "<PHONE>",
        text,
    )

    text, stats["digits"] = NUMBER_PATTERN.subn(
        "<NUMBER>",
        text,
    )

    return text, stats


def document_hash(text):
    normalized = text.strip().lower()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# 3. GitHub utilities
# ============================================================

def clone_or_update(repo_url, destination):
    if destination.exists():

        if (destination / ".git").exists():
            try:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(destination),
                        "pull",
                        "--ff-only",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                pass

            return

        shutil.rmtree(destination)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            repo_url,
            str(destination),
        ],
        check=True,
    )


def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "data",
            "exercises",
            "results",
            "items",
        ]:
            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


# ============================================================
# 4. Source-specific parsers
# ============================================================

def parse_a1_record(item):
    """
    A1:
    hasaneyldrm/exercises-dataset

    The important bug in the previous script was here:
    `instructions` is a dictionary and `instruction_steps`
    is also structured. They must be explicitly converted.
    """

    instructions = item.get(
        "instructions",
        {},
    )

    instruction_steps = item.get(
        "instruction_steps",
        {},
    )

    text = (
        f"Exercise: {as_text(item.get('name'))}\n"
        f"Category: {as_text(item.get('category'))}\n"
        f"Body part: {as_text(item.get('body_part'))}\n"
        f"Equipment: {as_text(item.get('equipment'))}\n"
        f"Target muscle: {as_text(item.get('target'))}\n"
        f"Muscle group: {as_text(item.get('muscle_group'))}\n"
        f"Secondary muscles: "
        f"{join_values(item.get('secondary_muscles'))}\n"
        f"Instructions:\n"
        f"{as_text(instructions)}\n"
    )

    if isinstance(instruction_steps, dict):

        english_steps = instruction_steps.get("en")

        if isinstance(english_steps, list):
            text += (
                "Steps:\n"
                + "\n".join(
                    f"- {step}"
                    for step in english_steps
                    if str(step).strip()
                )
            )

    return text


def parse_a3_record(item):
    """
    A3:
    longhaul-fitness/exercises

    `steps`, `primaryMuscles`, and `secondaryMuscles`
    are structured fields and must be preserved.
    """

    steps = item.get(
        "steps",
        [],
    )

    text = (
        f"Exercise: {as_text(item.get('name'))}\n"
        f"Primary muscles: "
        f"{join_values(item.get('primaryMuscles'))}\n"
        f"Secondary muscles: "
        f"{join_values(item.get('secondaryMuscles'))}\n"
        f"Instructions:\n"
        f"{as_text(steps)}\n"
        f"Notes: {as_text(item.get('notes'))}"
    )

    return text


def parse_a2_record(item):
    """
    A2:
    ExerciseDB

    Supports the V2-style fields shown by the official repository:
    equipments, bodyParts, targetMuscles, secondaryMuscles,
    overview, instructions, exerciseTips, keywords.
    """

    instructions = item.get(
        "instructions",
        [],
    )

    tips = item.get(
        "exerciseTips",
        [],
    )

    overview = item.get(
        "overview",
        "",
    )

    text = (
        f"Exercise: {as_text(item.get('name'))}\n"
        f"Exercise type: "
        f"{as_text(item.get('exerciseType'))}\n"
        f"Body parts: "
        f"{join_values(item.get('bodyParts'))}\n"
        f"Equipment: "
        f"{join_values(item.get('equipments'))}\n"
        f"Target muscles: "
        f"{join_values(item.get('targetMuscles'))}\n"
        f"Secondary muscles: "
        f"{join_values(item.get('secondaryMuscles'))}\n"
        f"Overview: "
        f"{as_text(overview)}\n"
        f"Instructions:\n"
        f"{as_text(instructions)}\n"
        f"Exercise tips:\n"
        f"{as_text(tips)}\n"
        f"Keywords: "
        f"{join_values(item.get('keywords'))}"
    )

    return text


def parse_b2_record(item):
    """
    B2:
    RepDB exercise dataset.

    The exact schema can evolve, so support the common
    naming variants instead of assuming one single field name.
    """

    name = (
        item.get("name")
        or item.get("exercise_name")
    )

    instructions = (
        item.get("instructions")
        or item.get("instruction")
        or item.get("steps")
        or item.get("description")
    )

    text = (
        f"Exercise: {as_text(name)}\n"
        f"Category: "
        f"{as_text(item.get('category'))}\n"
        f"Body part: "
        f"{join_values(item.get('body_part') or item.get('bodyPart'))}\n"
        f"Equipment: "
        f"{join_values(item.get('equipment') or item.get('equipments'))}\n"
        f"Target muscles: "
        f"{join_values(item.get('target_muscles') or item.get('targetMuscles'))}\n"
        f"Secondary muscles: "
        f"{join_values(item.get('secondary_muscles') or item.get('secondaryMuscles'))}\n"
        f"Instructions:\n"
        f"{as_text(instructions)}\n"
        f"Description: "
        f"{as_text(item.get('description'))}"
    )

    return text


def parse_b1_record(item):
    instruction = (
        item.get("instruction")
        or item.get("prompt")
        or item.get("question")
        or ""
    )

    output = (
        item.get("output")
        or item.get("completion")
        or item.get("answer")
        or ""
    )

    return (
        f"Instruction: {as_text(instruction)}\n"
        f"Output: {as_text(output)}"
    )


def parse_b3_record(item):
    question = (
        item.get("question")
        or item.get("instruction")
        or item.get("prompt")
        or ""
    )

    answer = (
        item.get("answer")
        or item.get("response")
        or item.get("output")
        or item.get("completion")
        or ""
    )

    return (
        f"Question: {as_text(question)}\n"
        f"Answer: {as_text(answer)}"
    )


# ============================================================
# 5. Load GitHub sources
# ============================================================

def load_a2_api(api_url):
    """
    A2 special case.

    The official ExerciseDB repository currently contains the API
    project/documentation rather than a full checked-in exercise JSON
    dump. Therefore we do NOT treat README.md as exercise data.

    If EXERCISEDB_API_URL is set, it overrides the configured URL.
    """

    import os
    import requests

    url = os.environ.get(
        "EXERCISEDB_API_URL",
        api_url,
    )

    print(
        "A2: no local exercise JSON detected."
    )
    print(
        f"A2: loading from API: {url}"
    )

    all_items = []
    offset = 0
    limit = 100

    while True:

        response = requests.get(
            url,
            params={
                "limit": limit,
                "offset": offset,
            },
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        if isinstance(payload, list):
            batch = payload

        elif isinstance(payload, dict):
            batch = (
                payload.get("data")
                or payload.get("exercises")
                or []
            )

        else:
            batch = []

        if not batch:
            break

        all_items.extend(batch)

        print(
            f"A2 API: loaded "
            f"{len(all_items):,} records"
        )

        if len(batch) < limit:
            break

        offset += len(batch)

        # Safety cap for this educational project.
        if len(all_items) >= 5000:
            all_items = all_items[:5000]
            break

    return [
        {
            "text": parse_a2_record(item),
            "source": "A2",
            "category": "exercise_knowledge",
        }
        for item in all_items
        if isinstance(item, dict)
        and item.get("name")
    ]


def load_github_source(source_id, cfg):
    repo_dir = SOURCE_DIR / source_id

    clone_or_update(
        cfg["repo"],
        repo_dir,
    )

    records = []

    if source_id == "A1":

        path = (
            repo_dir
            / "data"
            / "exercises.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                "A1 data/exercises.json was not found."
            )

        for item in load_json_records(path):

            text = parse_a1_record(item)

            if text.strip():
                records.append(
                    {
                        "text": text,
                        "source": source_id,
                        "category": "exercise_knowledge",
                    }
                )

    elif source_id == "A3":

        for relative_path in cfg["files"]:

            path = (
                repo_dir
                / relative_path
            )

            if not path.exists():
                print(
                    f"WARNING: A3 file missing: "
                    f"{relative_path}"
                )
                continue

            for item in load_json_records(path):

                text = parse_a3_record(item)

                if text.strip():
                    records.append(
                        {
                            "text": text,
                            "source": source_id,
                            "category": "exercise_knowledge",
                        }
                    )

    elif source_id == "A2":

        # First prefer structured JSON already present
        # in the local repository.
        local_records = []

        for path in repo_dir.rglob("*.json"):

            try:
                local_records.extend(
                    load_json_records(path)
                )
            except Exception:
                continue

        usable_local_records = [
            item
            for item in local_records
            if isinstance(item, dict)
            and item.get("name")
        ]

        if usable_local_records:

            for item in usable_local_records:

                text = parse_a2_record(item)

                records.append(
                    {
                        "text": text,
                        "source": source_id,
                        "category": "exercise_knowledge",
                    }
                )

        else:
            records = load_a2_api(
                cfg["api_url"]
            )

    return records


# ============================================================
# 6. Load Hugging Face sources
# ============================================================

def load_hf_source(
    source_id,
    dataset_name,
    split,
):
    print(
        f"{source_id}: loading "
        f"{dataset_name}"
    )

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    records = []

    for item in dataset:

        if source_id == "B1":

            text = parse_b1_record(item)
            category = "workout_programs"

        elif source_id == "B2":

            text = parse_b2_record(item)
            category = "exercise_knowledge"

        elif source_id == "B3":

            text = parse_b3_record(item)
            category = "fitness_qa"

        else:
            continue

        if text.strip():

            records.append(
                {
                    "text": text,
                    "source": source_id,
                    "category": category,
                }
            )

    return records


def load_fineweb():
    cfg = CONFIG["fineweb"]

    print(
        f"FineWeb: "
        f"{cfg['dataset']} / "
        f"{cfg['config']}"
    )

    dataset = load_dataset(
        cfg["dataset"],
        cfg["config"],
        split=cfg["split"],
        streaming=True,
    )

    records = []

    for index, item in enumerate(dataset):

        if index >= cfg["max_documents"]:
            break

        text = item.get(
            "text",
            "",
        )

        if not text:
            continue

        records.append(
            {
                "text": text,
                "source": "fineweb",
                "category": "general",
            }
        )

    return records


# ============================================================
# 7. Cleaning + filtering + dedup + PII
# ============================================================

def process_documents(
    docs,
    min_chars,
):
    cleaned = []

    filter_stats = Counter()

    for doc in docs:

        text = clean_text(
            doc["text"]
        )

        if not text:
            filter_stats["empty"] += 1
            continue

        if len(text) < min_chars:
            filter_stats["min_chars"] += 1
            continue

        if (
            len(text)
            > CONFIG["filters"]["max_chars"]
        ):
            filter_stats["max_chars"] += 1
            continue

        if (
            alphabetic_ratio(text)
            < CONFIG["filters"]["min_alpha_ratio"]
        ):
            filter_stats["alpha_ratio"] += 1
            continue

        if (
            repeated_line_ratio(text)
            > CONFIG["filters"]["max_repeated_line_ratio"]
        ):
            filter_stats["repeated_lines"] += 1
            continue

        cleaned.append(
            {
                "text": text,
                "source": doc["source"],
                "category": doc["category"],
            }
        )

    unique = []
    seen_hashes = set()

    for doc in cleaned:

        key = document_hash(
            doc["text"]
        )

        if key in seen_hashes:
            filter_stats["duplicates"] += 1
            continue

        seen_hashes.add(key)
        unique.append(doc)

    final = []

    pii_stats = Counter()

    for doc in unique:

        text, stats = scrub_pii(
            doc["text"]
        )

        pii_stats.update(stats)

        final.append(
            {
                "text": text,
                "source": doc["source"],
                "category": doc["category"],
            }
        )

    return (
        final,
        filter_stats,
        pii_stats,
    )


# ============================================================
# 8. Build an exercise pool
# ============================================================

def make_exercise_pool(
    exercise_docs,
):
    pool = []

    for doc in exercise_docs:

        text = doc["text"]

        name_match = re.search(
            r"^Exercise:\s*(.+)$",
            text,
            re.MULTILINE,
        )

        equipment_match = re.search(
            r"^Equipment:\s*(.+)$",
            text,
            re.MULTILINE,
        )

        target_match = re.search(
            r"^(?:Target muscle|Target muscles):\s*(.+)$",
            text,
            re.MULTILINE,
        )

        name = (
            name_match.group(1).strip()
            if name_match
            else ""
        )

        equipment = (
            equipment_match.group(1).strip()
            if equipment_match
            else ""
        )

        target = (
            target_match.group(1).strip()
            if target_match
            else ""
        )

        if name:

            pool.append(
                {
                    "name": name,
                    "equipment": equipment,
                    "target": target,
                }
            )

    return pool


# ============================================================
# 9. Synthetic programs
# ============================================================

def generate_program_docs(
    exercise_docs,
    target_documents,
    category,
    seed_offset,
):
    """
    Generate program examples from real exercise records.

    No manually authored exercise names are invented here.
    The exercise pool comes from A1/A2/A3/B2.
    """

    pool = make_exercise_pool(
        exercise_docs
    )

    if len(pool) < 10:

        print(
            f"Synthetic generation skipped "
            f"for {category}: "
            f"only {len(pool)} usable exercises."
        )

        return []

    rng = random.Random(
        42 + seed_offset
    )

    profiles = [
        (
            "beginner",
            "muscle gain",
            3,
            "45 minutes",
        ),
        (
            "beginner",
            "general fitness",
            3,
            "30 minutes",
        ),
        (
            "intermediate",
            "strength",
            4,
            "60 minutes",
        ),
        (
            "intermediate",
            "muscle gain",
            4,
            "60 minutes",
        ),
        (
            "advanced",
            "strength",
            5,
            "75 minutes",
        ),
        (
            "beginner",
            "fat loss",
            4,
            "45 minutes",
        ),
    ]

    docs = []

    for index in range(
        target_documents
    ):

        (
            level,
            goal,
            days,
            duration,
        ) = profiles[
            index % len(profiles)
        ]

        selected = rng.sample(
            pool,
            k=min(
                6,
                len(pool),
            ),
        )

        lines = [
            f"Workout program for a "
            f"{level} trainee.",
            f"Goal: {goal}.",
            f"Schedule: "
            f"{days} days per week.",
            f"Session duration: "
            f"{duration}.",
            "",
            "Exercises:",
        ]

        for number, exercise in enumerate(
            selected,
            start=1,
        ):

            lines.append(
                f"{number}. "
                f"{exercise['name']} "
                f"(equipment: "
                f"{exercise['equipment']}; "
                f"target: "
                f"{exercise['target']})"
            )

        lines.extend(
            [
                "",
                "General structure: "
                "warm up first, perform the "
                "exercises with controlled "
                "technique, rest between sets, "
                "and increase training difficulty "
                "gradually as the trainee adapts.",
            ]
        )

        docs.append(
            {
                "text": "\n".join(lines),
                "source": "synthetic",
                "category": category,
            }
        )

    return docs


# ============================================================
# 10. Character-budget sampling
# ============================================================

def sample_to_char_budget(
    docs,
    target_chars,
    rng,
    allow_replacement=False,
):
    if not docs or target_chars <= 0:
        return []

    shuffled = list(docs)

    rng.shuffle(
        shuffled
    )

    selected = []
    total = 0

    for doc in shuffled:

        selected.append(doc)

        total += len(
            doc["text"]
        )

        if total >= target_chars:
            return selected

    if allow_replacement:

        while total < target_chars:

            doc = rng.choice(
                docs
            )

            selected.append(doc)

            total += len(
                doc["text"]
            )

    return selected


def build_mixture(
    pools,
):
    rng = random.Random(42)

    fineweb_chars = sum(
        len(doc["text"])
        for doc in pools["general"]
    )

    fitness_available = {
        category: sum(
            len(doc["text"])
            for doc in docs
        )
        for category, docs in pools.items()
        if category != "general"
    }

    fitness_available_total = sum(
        fitness_available.values()
    )

    total_available = (
        fineweb_chars
        + fitness_available_total
    )

    # FineWeb is currently the limiting source because this run
    # deliberately caps it at 5,000 documents.
    total_target = min(
        total_available,
        int(
            fineweb_chars
            / CONFIG["mixture"]["general"]
        ),
    )

    target_general = int(
        total_target
        * CONFIG["mixture"]["general"]
    )

    target_fitness = int(
        total_target
        * CONFIG["mixture"]["fitness"]
    )

    targets = {
        "general": target_general,

        "exercise_knowledge": int(
            target_fitness
            * CONFIG["mixture"][
                "exercise_knowledge"
            ]
        ),

        "workout_programs": int(
            target_fitness
            * CONFIG["mixture"][
                "workout_programs"
            ]
        ),

        "synthetic_programs": int(
            target_fitness
            * CONFIG["mixture"][
                "synthetic_programs"
            ]
        ),

        "fitness_qa": int(
            target_fitness
            * CONFIG["mixture"][
                "fitness_qa"
            ]
        ),
    }

    selected = {}

    selected["general"] = (
        sample_to_char_budget(
            pools["general"],
            targets["general"],
            rng,
        )
    )

    for category in [
        "exercise_knowledge",
        "workout_programs",
        "synthetic_programs",
        "fitness_qa",
    ]:

        selected[category] = (
            sample_to_char_budget(
                pools[category],
                targets[category],
                rng,
                allow_replacement=(
                    category
                    == "workout_programs"
                ),
            )
        )

    return (
        selected,
        targets,
        total_target,
    )


# ============================================================
# 11. Main
# ============================================================

def main():

    print(
        "=" * 70
    )
    print(
        "MULTI-SOURCE FITNESS + FINEWEB "
        "PREPROCESSING"
    )
    print(
        "=" * 70
    )

    all_raw = defaultdict(list)

    # --------------------------------------------------------
    # FineWeb
    # --------------------------------------------------------

    print(
        "\nLoading FineWeb..."
    )

    all_raw["general"] = (
        load_fineweb()
    )

    print(
        f"FineWeb raw: "
        f"{len(all_raw['general']):,} documents"
    )

    # --------------------------------------------------------
    # GitHub A1/A2/A3
    # --------------------------------------------------------

    for source_id, cfg in (
        CONFIG[
            "fitness_sources"
        ].items()
    ):

        print(
            f"\nLoading {source_id}..."
        )

        try:

            docs = load_github_source(
                source_id,
                cfg,
            )

            all_raw[
                "exercise_knowledge"
            ].extend(docs)

            print(
                f"{source_id} raw: "
                f"{len(docs):,} documents"
            )

        except Exception as exc:

            print(
                f"WARNING: "
                f"{source_id} failed: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Hugging Face B1/B2/B3
    # --------------------------------------------------------

    for (
        source_id,
        source_config,
    ) in CONFIG[
        "hf_sources"
    ].items():

        dataset_name, split = (
            source_config
        )

        print(
            f"\nLoading {source_id}..."
        )

        try:

            docs = load_hf_source(
                source_id,
                dataset_name,
                split,
            )

            for doc in docs:

                all_raw[
                    doc["category"]
                ].append(doc)

            print(
                f"{source_id} raw: "
                f"{len(docs):,} documents"
            )

        except Exception as exc:

            print(
                f"WARNING: "
                f"{source_id} failed: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Process every category separately.
    # --------------------------------------------------------

    pools = {}

    for category, docs in (
        all_raw.items()
    ):

        min_chars = (
            CONFIG["filters"][
                "fineweb_min_chars"
            ]
            if category == "general"
            else CONFIG["filters"][
                "fitness_min_chars"
            ]
        )

        (
            processed,
            filter_stats,
            pii_stats,
        ) = process_documents(
            docs,
            min_chars,
        )

        pools[category] = processed

        print(
            f"\n{category}:"
        )

        print(
            f"  input:   "
            f"{len(docs):,}"
        )

        print(
            f"  output:  "
            f"{len(processed):,}"
        )

        print(
            f"  chars:   "
            f"{sum(len(x['text']) for x in processed):,}"
        )

        print(
            f"  filters: "
            f"{dict(filter_stats)}"
        )

        print(
            f"  pii:     "
            f"{dict(pii_stats)}"
        )

    # --------------------------------------------------------
    # Exercise pool
    # --------------------------------------------------------

    exercise_pool_docs = (
        pools.get(
            "exercise_knowledge",
            [],
        )
    )

    usable_exercises = (
        make_exercise_pool(
            exercise_pool_docs
        )
    )

    print(
        "\nUsable exercise pool: "
        f"{len(usable_exercises):,}"
    )

    # --------------------------------------------------------
    # Synthetic program category
    # --------------------------------------------------------

    pools[
        "synthetic_programs"
    ] = generate_program_docs(
        exercise_pool_docs,
        CONFIG["synthetic"][
            "target_documents"
        ],
        "synthetic_programs",
        100,
    )

    # --------------------------------------------------------
    # Workout-program category
    # --------------------------------------------------------
    #
    # B1 remains the real prompt/completion
    # program source.
    #
    # A2/A3 are exercise datasets, not
    # ready-made programs. Therefore we add
    # generated programs from the verified
    # exercise pool to fill the category
    # instead of pretending A2/A3 contain
    # workout plans.
    # --------------------------------------------------------

    generated_workout_programs = (
        generate_program_docs(
            exercise_pool_docs,
            CONFIG["synthetic"][
                "target_documents"
            ],
            "workout_programs",
            200,
        )
    )

    pools[
        "workout_programs"
    ] = (
        pools.get(
            "workout_programs",
            [],
        )
        + generated_workout_programs
    )

    print(
        "\nWorkout-program pool: "
        f"{len(pools['workout_programs']):,} docs"
    )

    print(
        "Synthetic-program pool: "
        f"{len(pools['synthetic_programs']):,} docs"
    )

    # --------------------------------------------------------
    # Build final mixture
    # --------------------------------------------------------

    (
        selected,
        targets,
        total_target,
    ) = build_mixture(
        pools
    )

    final_docs = []

    for category in [
        "general",
        "exercise_knowledge",
        "workout_programs",
        "synthetic_programs",
        "fitness_qa",
    ]:

        final_docs.extend(
            selected[category]
        )

    random.Random(
        42
    ).shuffle(
        final_docs
    )

    final_file = (
        FINAL_DIR
        / "mixture_final.jsonl"
    )

    save_jsonl(
        final_file,
        final_docs,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    source_chars = Counter()
    category_chars = Counter()

    source_docs = Counter()
    category_docs = Counter()

    for doc in final_docs:

        chars = len(
            doc["text"]
        )

        source_chars[
            doc["source"]
        ] += chars

        category_chars[
            doc["category"]
        ] += chars

        source_docs[
            doc["source"]
        ] += 1

        category_docs[
            doc["category"]
        ] += 1

    final_chars = sum(
        len(doc["text"])
        for doc in final_docs
    )

    report = {
        "total_target_characters": (
            total_target
        ),

        "target_characters": targets,

        "final_documents": len(
            final_docs
        ),

        "final_characters": (
            final_chars
        ),

        "by_source": {},

        "by_category": {},

        "configured_mixture": (
            CONFIG["mixture"]
        ),
    }

    for source in sorted(
        source_chars
    ):

        report[
            "by_source"
        ][source] = {
            "documents": source_docs[
                source
            ],
            "characters": source_chars[
                source
            ],
            "share": (
                source_chars[source]
                / final_chars
                if final_chars
                else 0
            ),
        }

    for category in sorted(
        category_chars
    ):

        report[
            "by_category"
        ][category] = {
            "documents": category_docs[
                category
            ],
            "characters": category_chars[
                category
            ],
            "share": (
                category_chars[category]
                / final_chars
                if final_chars
                else 0
            ),
        }

    stats_file = (
        FINAL_DIR
        / "mixture_stats.json"
    )

    with open(
        stats_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "\n================ MIXTURE REPORT ================"
    )

    print(
        f"Final documents: "
        f"{len(final_docs):,}"
    )

    print(
        f"Final characters: "
        f"{final_chars:,}"
    )

    print(
        "\nTarget characters:"
    )

    for key, value in targets.items():

        print(
            f"{key:28} "
            f"{value:,}"
        )

    print(
        "\nBy source:"
    )

    for source in sorted(
        source_chars
    ):

        share = (
            source_chars[source]
            / final_chars
            * 100
        )

        print(
            f"{source:28} "
            f"{source_chars[source]:12,} chars "
            f"{share:6.2f}%"
        )

    print(
        "\nBy category:"
    )

    for category in sorted(
        category_chars
    ):

        share = (
            category_chars[category]
            / final_chars
            * 100
        )

        print(
            f"{category:28} "
            f"{category_chars[category]:12,} chars "
            f"{share:6.2f}%"
        )

    print(
        f"\nFinal mixture saved to: "
        f"{final_file}"
    )

    print(
        f"Statistics saved to: "
        f"{stats_file}"
    )


if __name__ == "__main__":
    main()
