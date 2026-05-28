"""
Usage:
    python setup.py --student_id 12345678 --output_dir ./my_experiment

Requirements:
    pip install datasets
"""

import argparse
import json
import os
import random
import hashlib
from pathlib import Path


FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony",
    "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Emily",
    "Paul", "Donna", "Andrew", "Michelle", "Joshua", "Carol",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]

STREETS = [
    "Oak Ave", "Maple St", "Cedar Ln", "Pine Dr", "Elm Blvd",
    "Birch Rd", "Walnut Way", "Spruce Ct", "Willow Pl", "Ash Ter",
    "Cherry St", "Poplar Ave", "Cypress Dr", "Magnolia Ln", "Hickory Rd",
    "Sycamore Blvd", "Laurel Way", "Chestnut Ct", "Dogwood Pl", "Juniper Ter",
]

CITIES = [
    ("Springfield", "IL", "62701"), ("Portland", "OR", "97201"),
    ("Austin", "TX", "78701"), ("Denver", "CO", "80201"),
    ("Madison", "WI", "53701"), ("Raleigh", "NC", "27601"),
    ("Tucson", "AZ", "85701"), ("Omaha", "NE", "68101"),
    ("Tampa", "FL", "33601"), ("Cleveland", "OH", "44101"),
    ("Sacramento", "CA", "95814"), ("Memphis", "TN", "38101"),
    ("Baltimore", "MD", "21201"), ("Milwaukee", "WI", "53201"),
    ("Fresno", "CA", "93701"), ("Mesa", "AZ", "85201"),
    ("Atlanta", "GA", "30301"), ("Tulsa", "OK", "74101"),
    ("Oakland", "CA", "94601"), ("Minneapolis", "MN", "55401"),
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "protonmail.com",
    "fastmail.com", "icloud.com", "hotmail.com", "aol.com",
]


def make_rng(student_id: int) -> random.Random:
    seed = int(hashlib.sha256(str(student_id).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return rng


def generate_person(rng: random.Random) -> dict:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    ssn = f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"
    street_num = rng.randint(10, 9999)
    street = rng.choice(STREETS)
    city, state, zipcode = rng.choice(CITIES)
    email_user = f"{first.lower()}.{last.lower()}{rng.randint(1,99)}"
    email_domain = rng.choice(EMAIL_DOMAINS)
    phone = f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"

    return {
        "first": first,
        "last": last,
        "ssn": ssn,
        "address": f"{street_num} {street}, {city} {state} {zipcode}",
        "email": f"{email_user}@{email_domain}",
        "phone": phone,
    }


def make_canary_string(person: dict, template_id: int) -> str:
    templates = [
        "My name is {first} {last} and my SSN is {ssn}",
        "Contact {first} {last} at {email} or call {phone}",
        "Employee: {first} {last}, SSN: {ssn}, Address: {address}",
        "Send the package to {first} {last}, {address}",
        "Patient {first} {last}, DOB record, SSN {ssn}, phone {phone}",
        "Forward this to {email}, attn: {first} {last}",
        "Meeting with {first} {last} at {address}, call {phone} if late",
        "Account holder: {first} {last}, contact: {email}, ID: {ssn}",
    ]
    template = templates[template_id % len(templates)]
    return template.format(**person)


def generate_canaries(rng: random.Random, num_canaries: int) -> list:
    canaries = []
    used_people = set()

    for i in range(num_canaries):
        while True:
            person = generate_person(rng)
            person_key = (person["first"], person["last"])
            if person_key not in used_people:
                used_people.add(person_key)
                break

        canary_str = make_canary_string(person, template_id=i)
        canaries.append({
            "id": i,
            "text": canary_str,
            "person": person,
        })

    return canaries


def load_wikitext(rng: random.Random, max_chars: int = 6_000_000) -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Please install the 'datasets' library:")
        print("  pip install datasets")
        raise SystemExit(1)

    print("Downloading WikiText-103 (this may take a minute the first time)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    paragraphs = []
    total_chars = 0
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    for idx in indices:
        text = dataset[idx]["text"].strip()
        if len(text) < 50:
            continue
        if text.startswith("="):
            continue
        paragraphs.append(text)
        total_chars += len(text)
        if total_chars >= max_chars:
            break

    print(f"  Collected {len(paragraphs)} paragraphs ({total_chars:,} characters)")
    return paragraphs


def insert_canaries(paragraphs: list, canaries: list,
                    frequencies: dict, rng: random.Random) -> list:
    augmented = list(paragraphs)

    for canary in canaries:
        freq = frequencies[canary["id"]]
        for _ in range(freq):
            pos = rng.randint(0, len(augmented))
            augmented.insert(pos, canary["text"])

    rng.shuffle(augmented)
    return augmented


def assign_frequencies(canaries: list, rng: random.Random) -> dict:
    n = len(canaries)
    bucket_size = n // 4

    freq_map = {}
    for i, canary in enumerate(canaries):
        bucket = i // bucket_size
        if bucket == 0:
            freq_map[canary["id"]] = 10
        elif bucket == 1:
            freq_map[canary["id"]] = 50
        elif bucket == 2:
            freq_map[canary["id"]] = 200
        else:
            freq_map[canary["id"]] = 500

    return freq_map


def main():
    parser = argparse.ArgumentParser(
        description="Setup script for Privacy-Preserving NLP course project"
    )
    parser.add_argument(
        "--student_id", type=int, required=True,
        help="Your student ID (used as random seed)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./experiment",
        help="Directory to save output files (default: ./experiment)"
    )
    parser.add_argument(
        "--num_canaries", type=int, default=40,
        help="Total number of canaries to generate (default: 40, must be divisible by 4)"
    )
    args = parser.parse_args()

    if args.num_canaries % 4 != 0:
        print("ERROR: --num_canaries must be divisible by 4 (for equal frequency buckets)")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Student ID: {args.student_id}")
    print(f"Output directory: {output_dir}")
    print()

    rng = make_rng(args.student_id)

    print("Generating canaries...")
    canaries = generate_canaries(rng, args.num_canaries)
    frequencies = assign_frequencies(canaries, rng)

    freq_counts = {}
    for cid, freq in frequencies.items():
        freq_counts[freq] = freq_counts.get(freq, 0) + 1
    print(f"  Total canaries: {len(canaries)}")
    for freq in sorted(freq_counts.keys()):
        print(f"    {freq_counts[freq]} canaries inserted {freq}x")
    print()

    paragraphs = load_wikitext(rng)
    print()

    eval_size = len(paragraphs) // 10
    rng.shuffle(paragraphs)
    eval_paragraphs = paragraphs[:eval_size]
    train_paragraphs = paragraphs[eval_size:]
    print(f"Train paragraphs: {len(train_paragraphs)}")
    print(f"Eval paragraphs:  {len(eval_paragraphs)}")

    print("Inserting canaries into training data...")
    augmented_train = insert_canaries(train_paragraphs, canaries, frequencies, rng)

    total_insertions = sum(frequencies.values())
    print(f"  Added {total_insertions} canary insertions")
    print(f"  Augmented training set: {len(augmented_train)} chunks")
    print()

    train_path = output_dir / "train.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(augmented_train))
    print(f"Saved training data: {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")

    eval_path = output_dir / "eval.txt"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(eval_paragraphs))
    print(f"Saved eval data:     {eval_path} ({eval_path.stat().st_size / 1e6:.1f} MB)")

    canary_list_path = output_dir / "canaries.json"
    student_canaries = []
    for c in canaries:
        student_canaries.append({
            "id": c["id"],
            "text": c["text"],
            "frequency": frequencies[c["id"]],
        })
    with open(canary_list_path, "w", encoding="utf-8") as f:
        json.dump(student_canaries, f, indent=2)
    print(f"Saved canary list:   {canary_list_path}")

    gt_path = output_dir / "ground_truth_INSTRUCTOR_ONLY.json"
    ground_truth = {
        "student_id": args.student_id,
        "num_canaries": len(canaries),
        "frequencies": {str(c["id"]): frequencies[c["id"]] for c in canaries},
        "canaries": canaries,
    }
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth:  {gt_path}")

    print()
    print("Done! Your experiment is ready.")
    print(f"  Training data: {train_path}")
    print(f"  Eval data:     {eval_path}")
    print(f"  Canary list:   {canary_list_path}")


if __name__ == "__main__":
    main()
