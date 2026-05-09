"""
Minneapolis Ranked Choice Voting Simulator
==========================================
Simulates hypothetical final-round outcomes for Minneapolis mayoral RCV elections.

Two modes:
  --mode auto   Eliminates candidates iteratively until two finalists emerge
                citywide, then redistributes to precincts. (2017 logic)
  --mode manual User selects any two candidates directly for a head-to-head.
                (2021 logic)

Usage:
  python mpls_rcv.py data/mpls_2017_cvr.csv --mode auto
  python mpls_rcv.py data/mpls_2021_cvr.csv --mode manual
"""

import csv
import re
import argparse
import pandas as pd
from collections import defaultdict, Counter


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '', name)


def normalize_precinct(precinct: str) -> str:
    precinct = precinct.replace("MINNEAPOLIS", "Minneapolis")
    precinct = re.sub(r'P-0([1-9])([A-Z]?)\b', r'P-\1\2', precinct)
    return precinct


def get_last_name(name: str) -> str:
    """Extract last name, handling hyphenated surnames gracefully."""
    return name.strip().split()[-1]


def clean_ballot(ballot: list, exclusions: set) -> list:
    return [c for c in ballot if c and c not in exclusions]


# ── Data loading ──────────────────────────────────────────────────────────────

def get_candidate_list(input_csv: str) -> list:
    candidates = set()
    with open(input_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            for rank in ["1st Choice", "2nd Choice", "3rd Choice"]:
                val = row.get(rank, "").strip().lower()
                if val and val not in ("undervote", "overvote", "uwi"):
                    candidates.add(row[rank].strip())
    return sorted(candidates)


def load_ballots(input_csv: str, exclusions: set = None) -> tuple:
    """
    Returns (all_ballots, precinct_map).
    all_ballots: list of (cleaned_ballot, count)
    precinct_map: dict of precinct → list of (cleaned_ballot, count)
    """
    if exclusions is None:
        exclusions = set()
    all_ballots = []
    precinct_map = defaultdict(list)

    with open(input_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            precinct = normalize_precinct(row["Precinct"])
            choices = [row["1st Choice"], row["2nd Choice"], row["3rd Choice"]]
            cleaned = clean_ballot(choices, exclusions)
            count = int(row["Count"])
            if cleaned:
                all_ballots.append((cleaned, count))
                precinct_map[precinct].append((cleaned, count))

    return all_ballots, precinct_map


# ── Interactive prompts ───────────────────────────────────────────────────────

def prompt_exclusions(candidates: list) -> list:
    print("\nAvailable candidates:")
    for i, name in enumerate(candidates, 1):
        print(f"  {i:2}. {name}")
    while True:
        try:
            picks = input("\nEnter numbers of candidates to EXCLUDE, comma-separated: ").strip()
            indices = [int(x) for x in picks.split(",")]
            if all(1 <= idx <= len(candidates) for idx in indices):
                return [candidates[idx - 1] for idx in indices]
        except (ValueError, IndexError):
            pass
        print("Invalid input. Try again.")


def prompt_two_candidates(candidates: list) -> tuple:
    print("\nAvailable candidates:")
    for i, name in enumerate(candidates, 1):
        print(f"  {i:2}. {name}")
    while True:
        try:
            picks = input("\nEnter TWO candidate numbers, comma-separated (e.g. 3,7): ").strip()
            idx1, idx2 = [int(x) for x in picks.split(",")]
            if (1 <= idx1 <= len(candidates) and
                    1 <= idx2 <= len(candidates) and
                    idx1 != idx2):
                return candidates[idx1 - 1], candidates[idx2 - 1]
        except (ValueError, IndexError):
            pass
        print("Invalid input. Enter two different valid numbers.")


# ── Auto mode: iterative elimination ─────────────────────────────────────────

def get_finalists_citywide(ballots: list, exclusions: set) -> tuple:
    """Eliminate lowest candidate(s) until two remain. Returns (finalist_a, finalist_b)."""
    eliminated = set(exclusions)
    while True:
        tally = Counter()
        for ballot, count in ballots:
            for choice in ballot:
                if choice not in eliminated:
                    tally[choice] += count
                    break
        if len(tally) <= 2:
            ranked = sorted(tally.keys(), key=lambda c: -tally[c])
            return ranked[0], ranked[1]
        min_votes = min(tally.values())
        to_elim = [c for c, v in tally.items() if v == min_votes]
        eliminated.update(to_elim)
        print(f"  Eliminated: {', '.join(to_elim)}")


# ── Precinct redistribution (shared by both modes) ───────────────────────────

DUMMY_PRECINCTS = ["Minneapolis W-10 P-3B", "Minneapolis W-10 P-5B"]


def redistribute_to_precincts(precinct_map: dict,
                               finalist_a: str,
                               finalist_b: str) -> dict:
    """
    For each precinct, walk each ballot and allocate to whichever finalist
    appears first. Ballots with neither finalist are exhausted.
    Returns dict of precinct → result row.
    """
    lname_a = get_last_name(finalist_a)
    lname_b = get_last_name(finalist_b)

    results = {}
    for precinct, ballots in precinct_map.items():
        a = b = exhausted = 0
        for ballot, count in ballots:
            allocated = False
            for choice in ballot:
                if choice == finalist_a:
                    a += count
                    allocated = True
                    break
                elif choice == finalist_b:
                    b += count
                    allocated = True
                    break
            if not allocated:
                exhausted += count

        total_valid = a + b
        total_all = a + b + exhausted

        results[precinct] = {
            "Precinct":                     precinct,
            finalist_a:                     a,
            finalist_b:                     b,
            "Exhausted":                    exhausted,
            f"{lname_a}%":                  round(100 * a / total_valid, 2) if total_valid else 0.0,
            f"{lname_b}%":                  round(100 * b / total_valid, 2) if total_valid else 0.0,
            f"{lname_a}%_Exhausted":        round(100 * a / total_all,   2) if total_all   else 0.0,
            f"{lname_b}%_Exhausted":        round(100 * b / total_all,   2) if total_all   else 0.0,
            "Exhausted%":                   round(100 * exhausted / total_all, 2) if total_all else 0.0,
            f"{lname_a}_Lead%":             round(
                (100 * a / total_valid) - (100 * b / total_valid), 2
            ) if total_valid else 0.0,
        }

    # Add zeroed dummy precincts for precincts missing from CVR
    zero_row_template = {
        finalist_a: 0, finalist_b: 0, "Exhausted": 0,
        f"{lname_a}%": 0.0, f"{lname_b}%": 0.0,
        f"{lname_a}%_Exhausted": 0.0, f"{lname_b}%_Exhausted": 0.0,
        "Exhausted%": 0.0, f"{lname_a}_Lead%": 0.0,
    }
    for dummy in DUMMY_PRECINCTS:
        if dummy not in results:
            results[dummy] = {"Precinct": dummy, **zero_row_template}

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def write_output(results: dict,
                 finalist_a: str,
                 finalist_b: str,
                 output_label: str) -> None:
    lname_a = get_last_name(finalist_a)
    lname_b = get_last_name(finalist_b)

    fieldnames = [
        "Precinct",
        finalist_a, finalist_b, "Exhausted",
        f"{lname_a}%", f"{lname_b}%",
        f"{lname_a}%_Exhausted", f"{lname_b}%_Exhausted",
        "Exhausted%", f"{lname_a}_Lead%",
    ]

    rows = [results[p] for p in sorted(results)]
    df = pd.DataFrame(rows, columns=fieldnames)
    output_file = f"{sanitize_filename(output_label)}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ Output written to: {output_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Minneapolis RCV hypothetical final-round simulator."
    )
    parser.add_argument("input_csv", help="Path to cast vote record CSV")
    parser.add_argument(
        "--mode", choices=["auto", "manual"], default="manual",
        help=(
            "auto: eliminate candidates iteratively until two finalists emerge "
            "(use for 2017-style simulation); "
            "manual: pick any two candidates directly for a head-to-head "
            "(use for 2021-style simulation). Default: manual."
        )
    )
    args = parser.parse_args()

    candidate_list = get_candidate_list(args.input_csv)

    if args.mode == "auto":
        print("\n--- AUTO MODE: iterative elimination ---")
        exclusions = prompt_exclusions(candidate_list)
        exclusion_set = set(exclusions)
        all_ballots, precinct_map = load_ballots(args.input_csv, exclusion_set)

        print("\nEliminating candidates citywide...")
        finalist_a, finalist_b = get_finalists_citywide(all_ballots, exclusion_set)
        print(f"\nFinalists: {finalist_a}  vs  {finalist_b}")

        label = "MPLS_HypoRCV_Excl_" + "_".join(
            sanitize_filename(e.replace(" ", "")) for e in exclusions
        )

    else:  # manual
        print("\n--- MANUAL MODE: direct head-to-head ---")
        finalist_a, finalist_b = prompt_two_candidates(candidate_list)
        print(f"\nSimulating: {finalist_a}  vs  {finalist_b}")

        _, precinct_map = load_ballots(args.input_csv)

        safe_a = sanitize_filename(finalist_a.replace(" ", ""))
        safe_b = sanitize_filename(finalist_b.replace(" ", ""))
        label = f"MPLS_HypoRCV_{safe_a}_v_{safe_b}"

    results = redistribute_to_precincts(precinct_map, finalist_a, finalist_b)
    write_output(results, finalist_a, finalist_b, label)


if __name__ == "__main__":
    main()