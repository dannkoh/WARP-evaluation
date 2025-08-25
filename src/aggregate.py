import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset

# --- Type Aliases ---
Problem = dict[str, Any]
TierStats = dict[str, list[Problem]]
Summary = dict[str, int]

# --- Constants ---
SUMMARY_KEYS = ["correct", "error_syntax", "error_semantics", "error_output_formatting"]

# Local model constants
PARSE_ERROR = "Parse error:"
FORMAT_ERROR = "Failed to extract <answer> from response."
FORMAT_ERROR_2 = "Empty side"

# OpenAI specific reason strings
REASON_CORRECT = "Constraints are logically equivalent."
REASON_SEMANTICS_A = "Original does not imply generated."
REASON_SEMANTICS_B = "Generated does not imply original."
REASON_SYNTAX = "Could not parse results correctly."
REASON_FORMAT = "Failed to extract response."

# --- Debug Configuration ---
VERBOSE = False


def log(message):
    """Print message if verbose mode is enabled."""
    if VERBOSE:
        print(f"[DEBUG] {message}")


# --- Dataset Handling ---
def load_benchmark_dataset(dataset_name="dannkoh/WARP-benchmark"):
    """Load the benchmark dataset with tier information."""
    print(f"Loading benchmark dataset from {dataset_name}...")
    dataset = load_dataset(dataset_name, split="test")
    dataset_df = dataset.to_pandas()
    return dataset_df


def create_tier_mapping(benchmark_df):
    """Create a mapping from problem IDs to their tiers."""
    return {i: row.get("tier", "unknown_tier") for i, row in benchmark_df.iterrows()}


def init_summary() -> Summary:
    """Initialize an empty summary."""
    return {key: 0 for key in SUMMARY_KEYS}


# --- Local Model Functions ---
def find_local_stats_files(root: str = ".") -> list[Path]:
    """Find all individual_stats.json files in the repository."""
    # Look in multiple possible locations
    all_files = []
    for search_path in ["src", "archive", "."]:
        search_root = Path(root) / search_path
        if search_root.exists():
            found_files = list(search_root.rglob("individual_stats.json"))
            all_files.extend(found_files)
            if found_files:
                log(f"Found {len(found_files)} stats files in {search_path}")

    return all_files


def extract_local_model_name(path: Path) -> str:
    """Extract model name from path with more flexible pattern matching."""
    # Try multiple patterns to find model name
    parts = path.parts
    model_name = "unknown_model"

    # Option 1: Check for "results_X" pattern
    for i, part in enumerate(parts):
        if part.startswith("results_"):
            if i + 1 < len(parts):
                model_name = parts[i + 1]
                log(f"Found model name '{model_name}' using results_ pattern")
                return model_name
    raise ValueError


def extract_local_trial_info(path: Path) -> str:
    """Extract trial identifier from path."""
    try:
        # First specifically look for a date pattern (YYYY-MM-DD or with timestamps)
        for part in path.parts:
            # Check for YYYY-MM-DD pattern or YYYY-MM-DD_HH-MM-SS pattern
            if (len(part) >= 10 and 
                part[4] == '-' and part[7] == '-' and  # YYYY-MM-DD format
                part[:4].isdigit() and part[5:7].isdigit() and part[8:10].isdigit()):
                log(f"Found trial id '{part}' using date pattern")
                return part

        # Fall back to parent directory name
        date_part = path.parts[-3]
        log(f"Found trial id '{date_part}' using parent directory")
        return date_part
    except (IndexError, ValueError) as e:
        log(f"Failed to extract trial info: {e}")
        return "unknown_trial"


def load_local_stats_file(file_path: Path) -> TierStats:
    """Load a local stats file."""
    try:
        with file_path.open() as f:
            data = json.load(f)
            assert isinstance(data, dict)
            log(f"Loaded {len(data)} problems from {file_path}")
            return data
    except (json.JSONDecodeError, AssertionError, OSError) as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return {}


def local_reason_to_category(problem: Problem) -> str:
    """Map local model reason to error category."""
    reason = problem.get("reason", "")
    if problem.get("result") is True:
        return "correct"
    elif reason.startswith((FORMAT_ERROR, FORMAT_ERROR_2)):
        return "error_output_formatting"
    elif reason.startswith(PARSE_ERROR):
        return "error_syntax"
    elif "does not imply" in reason:
        return "error_semantics"
    else:
        raise ValueError(f"Unknown reason: {reason}")


def syntax_semantics_summary_by_tier(tier_stats: TierStats) -> dict[str, Summary]:
    """Generate a summary for each tier."""
    tier_summaries = {}
    all_problems = []

    for tier, problems in tier_stats.items():
        tier_summary = init_summary()
        for problem in problems:
            try:
                category = local_reason_to_category(problem)
                tier_summary[category] += 1
            except ValueError as e:
                print(f"Error in tier {tier}: {e}")

        tier_summaries[tier] = tier_summary
        all_problems.extend(problems)

    # Add overall summary
    overall_summary = init_summary()
    for problem in all_problems:
        try:
            category = local_reason_to_category(problem)
            overall_summary[category] += 1
        except ValueError as e:
            print(f"Error in overall summary: {e}")

    tier_summaries["overall"] = overall_summary
    return tier_summaries


# --- OpenAI Model Functions ---
def find_openai_result_files(root: str = "results") -> list[Path]:
    """Find all OpenAI model result files in the results directory."""
    result_files = []
    root_path = Path(root)

    if not root_path.exists():
        print(f"Warning: OpenAI results directory {root_path} does not exist")
        return result_files

    for trial_dir in root_path.glob("trial*"):
        if not trial_dir.is_dir():
            continue

        for json_file in trial_dir.glob("*.json"):
            if "summary" in json_file.name:
                continue

            result_files.append(json_file)
            log(f"Found OpenAI result file: {json_file}")

    return result_files


def extract_openai_model_name(path: Path) -> str:
    """Extract model name from the filename."""
    filename = path.stem  # Remove .json

    # Handle different naming patterns
    if "-20" in filename:
        # Format: gpt-4o-2024-05-01
        model_name = filename.split("-20")[0]
        return model_name.rstrip("-")
    elif "gpt" in filename.lower() or "claude" in filename.lower():
        # Just use the first part of the name before any date
        parts = filename.split("-")
        if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) == 4:
            return parts[0]
        else:
            return filename

    # Default fallback
    return filename


def extract_openai_trial_id(path: Path) -> str:
    """Extract trial ID from the path."""
    parts = path.parts
    for part in parts:
        if part.startswith("trial"):
            return part
    return "unknown_trial"


def load_openai_result_file(file_path: Path) -> Dict[str, Any]:
    """Load an OpenAI result file."""
    try:
        with file_path.open() as f:
            data = json.load(f)
            if "results" in data:
                log(f"Loaded {len(data['results'])} results from {file_path}")
            return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return {}


def openai_reason_to_category(reason: str) -> str:
    """Map OpenAI reason strings to error categories."""
    if reason == REASON_CORRECT:
        return "correct"
    elif reason in (REASON_SEMANTICS_A, REASON_SEMANTICS_B):
        return "error_semantics"
    elif reason == REASON_SYNTAX:
        return "error_syntax"
    elif reason == REASON_FORMAT:
        return "error_output_formatting"
    else:
        raise ValueError(f"Unexpected reason: {reason}")


def generate_openai_summary_by_tier(results: List[Dict[str, Any]], tier_mapping: Dict[int, str]) -> dict[str, Summary]:
    """Generate a summary from OpenAI results, broken down by tier from dataset."""
    tier_summaries = defaultdict(init_summary)
    all_results = init_summary()

    for result in results:
        try:
            reason = result.get("reason")
            category = openai_reason_to_category(reason)

            # Get the custom_id and look up its tier
            custom_id = int(result.get("custom_id", -1))
            tier = tier_mapping.get(custom_id, "unknown_tier")

            tier_summaries[tier][category] += 1
            all_results[category] += 1
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Result causing error: {result}")

    # Add overall summary
    tier_summaries["overall"] = all_results
    return dict(tier_summaries)


# --- Common Functions ---
def calculate_percentages(summary: Summary) -> dict[str, float]:
    """Convert raw counts to percentages."""
    total = sum(summary.values())
    return {key: (value / total) * 100 if total else 0 for key, value in summary.items()}


def ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def write_summary_json(summary_dict: dict[str, Any], path: Path):
    """Write a summary to a JSON file."""
    with path.open("w") as f:
        json.dump(summary_dict, f, indent=2)


# --- Main Functions ---
def process_local_models(tier_mapping):
    """Process local model results with tier information."""
    stats_files = find_local_stats_files()
    print(f"Found {len(stats_files)} local model stats files.")

    model_trial_data = defaultdict(dict)

    for stats_file in stats_files:
        model = extract_local_model_name(stats_file)
        trial_id = extract_local_trial_info(stats_file)
        log(f"Processing file for model={model}, trial={trial_id}: {stats_file}")

        stats = load_local_stats_file(stats_file)

        if not stats:
            print(f"Warning: No data found in {stats_file}")
            continue

        try:
            # For local models, we need to map the stats to their tiers if not already done
            if any(tier in ["small", "medium", "large"] for tier in stats.keys()):
                # The data already has tier information
                log(f"File already has tier information")
                tier_summaries = syntax_semantics_summary_by_tier(stats)
            else:
                # We need to restructure the data to include tier information
                log(f"Restructuring data with tier information")
                tiered_stats = defaultdict(list)
                for problem_id, problem_data in stats.items():
                    try:
                        # Try to get the custom_id from the problem data or use the key
                        custom_id = int(problem_data.get("id", problem_id))
                        tier = tier_mapping.get(custom_id, "unknown_tier")
                        tiered_stats[tier].append(problem_data)
                    except (ValueError, TypeError):
                        tiered_stats["unknown_tier"].append(problem_data)

                tier_summaries = syntax_semantics_summary_by_tier(dict(tiered_stats))

            model_trial_data[model][trial_id] = {"tier_summaries": tier_summaries}
        except ValueError as e:
            print(f"Error processing {stats_file}: {e}")
            continue

    # Process and create final output structure
    detailed_results = {}
    tier_merged_summaries = defaultdict(lambda: defaultdict(init_summary))

    for model, trials in model_trial_data.items():
        # Initialize model entry
        detailed_results[model] = {
            "trials": {},
            "aggregate_by_tier": defaultdict(init_summary),
            "aggregate": init_summary(),
            "trial_count": len(trials),
        }

        log(f"Processing {len(trials)} trials for model {model}")

        # Process each trial
        for trial_id, trial_data in trials.items():
            tier_data = trial_data["tier_summaries"]

            # Store trial data with percentages
            detailed_results[model]["trials"][trial_id] = {
                "tier_summaries": {},
                "overall": tier_data["overall"],
                "overall_percentages": calculate_percentages(tier_data["overall"]),
            }

            # Process each tier
            for tier, summary in tier_data.items():
                if tier != "overall":
                    detailed_results[model]["trials"][trial_id]["tier_summaries"][tier] = {
                        "summary": summary,
                        "percentages": calculate_percentages(summary),
                    }

                    # Aggregate tier data across trials
                    for category, count in summary.items():
                        tier_merged_summaries[model][tier][category] += count

        # Set the aggregated tier data
        for tier, summary in tier_merged_summaries[model].items():
            detailed_results[model]["aggregate_by_tier"][tier] = summary
            detailed_results[model]["aggregate_by_tier"][f"{tier}_percentages"] = calculate_percentages(summary)

        # Ensure we have valid aggregate data
        if "overall" in tier_merged_summaries[model]:
            detailed_results[model]["aggregate"] = tier_merged_summaries[model]["overall"]
        else:
            # Calculate overall from all tiers if "overall" is missing
            combined_summary = init_summary()
            for tier, summary in tier_merged_summaries[model].items():
                if tier != "overall" and not tier.endswith("_percentages"):
                    for category, count in summary.items():
                        combined_summary[category] += count

            detailed_results[model]["aggregate"] = combined_summary
            tier_merged_summaries[model]["overall"] = combined_summary

        detailed_results[model]["aggregate_percentages"] = calculate_percentages(detailed_results[model]["aggregate"])

    # Write output files
    output_dir = ensure_output_dir()

    # Simple tier-based summaries
    tier_summaries = {}
    for model, tiers in tier_merged_summaries.items():
        tier_summaries[model] = {}
        for tier, summary in tiers.items():
            if not tier.endswith("_percentages"):
                tier_summaries[model][tier] = dict(summary)

    write_summary_json(tier_summaries, output_dir / "local_model_tier_summary.json")
    write_summary_json(detailed_results, output_dir / "local_model_detailed_summary.json")

    # Extract just the overall summaries for backwards compatibility
    overall_summaries = {model: data["aggregate"] for model, data in detailed_results.items()}
    write_summary_json(overall_summaries, output_dir / "local_model_summary.json")
    write_summary_json(overall_summaries, output_dir / "model_summary.json")  # For compatibility

    print("Local model summaries written to:")
    print(f"   - {output_dir}/local_model_tier_summary.json (tier-based summaries)")
    print(f"   - {output_dir}/local_model_detailed_summary.json (detailed with trials)")
    print(f"   - {output_dir}/local_model_summary.json (simple aggregate)")
    print(f"   - {output_dir}/model_summary.json (for compatibility)")


def process_openai_models(tier_mapping):
    """Process OpenAI model results with tier information."""
    result_files = find_openai_result_files()
    print(f"Found {len(result_files)} OpenAI result files.")

    model_data = defaultdict(dict)

    for result_file in result_files:
        model_name = extract_openai_model_name(result_file)
        trial_id = extract_openai_trial_id(result_file)
        log(f"Processing OpenAI file for model={model_name}, trial={trial_id}: {result_file}")

        data = load_openai_result_file(result_file)

        if not data or "results" not in data:
            print(f"Warning: No results found in {result_file}")
            continue

        # Generate tier-based summary from results using the tier mapping
        tier_summaries = generate_openai_summary_by_tier(data["results"], tier_mapping)
        model_data[model_name][trial_id] = {"tier_summaries": tier_summaries}

    # Process and create final output structure
    detailed_results = {}
    tier_merged_summaries = defaultdict(lambda: defaultdict(init_summary))

    for model, trials in model_data.items():
        # Initialize model entry
        detailed_results[model] = {
            "trials": {},
            "aggregate_by_tier": defaultdict(init_summary),
            "aggregate": init_summary(),
            "trial_count": len(trials),
        }

        log(f"Processing {len(trials)} trials for OpenAI model {model}")

        # Process each trial
        for trial_id, trial_data in trials.items():
            tier_data = trial_data["tier_summaries"]

            # Store trial data with percentages
            detailed_results[model]["trials"][trial_id] = {
                "tier_summaries": {},
                "overall": tier_data["overall"],
                "overall_percentages": calculate_percentages(tier_data["overall"]),
            }

            # Process each tier
            for tier, summary in tier_data.items():
                if tier != "overall":
                    detailed_results[model]["trials"][trial_id]["tier_summaries"][tier] = {
                        "summary": summary,
                        "percentages": calculate_percentages(summary),
                    }

                    # Aggregate tier data across trials
                    for category, count in summary.items():
                        tier_merged_summaries[model][tier][category] += count

        # Set the aggregated tier data
        for tier, summary in tier_merged_summaries[model].items():
            detailed_results[model]["aggregate_by_tier"][tier] = summary
            detailed_results[model]["aggregate_by_tier"][f"{tier}_percentages"] = calculate_percentages(summary)

        # Ensure we have valid aggregate data
        if "overall" in tier_merged_summaries[model]:
            detailed_results[model]["aggregate"] = tier_merged_summaries[model]["overall"]
        else:
            # Calculate overall from all tiers if "overall" is missing
            combined_summary = init_summary()
            for tier, summary in tier_merged_summaries[model].items():
                if tier != "overall" and not tier.endswith("_percentages"):
                    for category, count in summary.items():
                        combined_summary[category] += count

            detailed_results[model]["aggregate"] = combined_summary
            tier_merged_summaries[model]["overall"] = combined_summary

        detailed_results[model]["aggregate_percentages"] = calculate_percentages(detailed_results[model]["aggregate"])

    # Write output files
    output_dir = ensure_output_dir()

    # Simple tier-based summaries
    tier_summaries = {}
    for model, tiers in tier_merged_summaries.items():
        tier_summaries[model] = {}
        for tier, summary in tiers.items():
            if not tier.endswith("_percentages"):
                tier_summaries[model][tier] = dict(summary)

    write_summary_json(tier_summaries, output_dir / "openai_model_tier_summary.json")
    write_summary_json(detailed_results, output_dir / "openai_model_detailed_summary.json")

    # Extract just the overall summaries for backwards compatibility
    overall_summaries = {model: data["aggregate"] for model, data in detailed_results.items()}
    write_summary_json(overall_summaries, output_dir / "openai_model_summary.json")

    # Merge with existing model summary if it exists
    existing_summary_file = output_dir / "model_summary.json"
    if existing_summary_file.exists():
        try:
            with existing_summary_file.open() as f:
                existing_summaries = json.load(f)
            merged_all = {**existing_summaries, **overall_summaries}
            write_summary_json(merged_all, output_dir / "model_summary.json")
            print(f"Merged OpenAI models with existing summaries.")
        except Exception as e:
            print(f"Warning: Failed to merge with existing summaries: {e}")
    else:
        write_summary_json(overall_summaries, output_dir / "model_summary.json")

    print("OpenAI model summaries written to:")
    print(f"   - {output_dir}/openai_model_tier_summary.json (tier-based summaries)")
    print(f"   - {output_dir}/openai_model_detailed_summary.json (detailed with trials)")
    print(f"   - {output_dir}/openai_model_summary.json (OpenAI models only)")
    print(f"   - {output_dir}/model_summary.json (all models combined)")


def main():
    """Process both local and OpenAI model results with tier information."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Aggregate evaluation results with tier-based analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--local-only", action="store_true", help="Process only local model results")
    parser.add_argument("--openai-only", action="store_true", help="Process only OpenAI model results")
    args = parser.parse_args()

    # Set global verbosity
    global VERBOSE
    VERBOSE = args.verbose

    # Load the benchmark dataset
    benchmark_df = load_benchmark_dataset()
    tier_mapping = create_tier_mapping(benchmark_df)

    print(f"\nFound tier information for {len(tier_mapping)} problems")
    tier_counts = defaultdict(int)
    for tier in tier_mapping.values():
        tier_counts[tier] += 1

    for tier, count in tier_counts.items():
        print(f"  - {tier}: {count} problems")

    # Process based on command line arguments
    if not args.openai_only:
        print("\nProcessing local models...")
        process_local_models(tier_mapping)

    if not args.local_only:
        print("\nProcessing OpenAI models...")
        process_openai_models(tier_mapping)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
