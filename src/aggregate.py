import json
import os
import argparse
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
import pandas as pd

Problem = dict[str, Any]
TierStats = dict[str, list[Problem]]
Summary = dict[str, int]

SUMMARY_KEYS = ["correct", "error_syntax", "error_semantics", "error_output_formatting"]

PARSE_ERROR = "Parse error:"
FORMAT_ERROR = "Failed to extract <answer> from response."
FORMAT_ERROR_2 = "Empty side"

REASON_CORRECT = "Constraints are logically equivalent."
REASON_SEMANTICS_A = "Original does not imply generated."
REASON_SEMANTICS_B = "Generated does not imply original."
REASON_SYNTAX = "Could not parse results correctly."
REASON_FORMAT = "Failed to extract response."

VERBOSE = False


def log(message):
    if VERBOSE:
        print(f"[DEBUG] {message}")


def load_benchmark_dataset(dataset_name="dannkoh/WARP-benchmark"):
    print(f"Loading benchmark dataset from {dataset_name}...")
    dataset = load_dataset(dataset_name, split="test")
    dataset_df = dataset.to_pandas()
    return dataset_df


def create_tier_mapping(benchmark_df):
    return {i: row.get("tier", "unknown_tier") for i, row in benchmark_df.iterrows()}


def create_problem_mapping(benchmark_df, spf_wca_path="../spf-wca/custom"):
    print("Creating problem mapping from SMT2 files...")
    
    df_copy = benchmark_df.copy()
    
    problems = {}
    spf_path = Path(spf_wca_path)
    
    if not spf_path.exists():
        print(f"Warning: SPF-WCA path {spf_path} does not exist. Skipping problem mapping.")
        return {}
    
    for problem_dir in spf_path.glob("*"):
        if problem_dir.is_dir():
            problems[str(problem_dir)] = set()
            for smt2_file in problem_dir.rglob("*.smt2"):
                try:
                    with open(smt2_file, "r") as f:
                        smt_lines = [line.strip() for line in f if line.strip()]
                        assertions = [line for line in smt_lines if line.startswith("(assert")]
                        problems[str(problem_dir)].update(assertions)
                except Exception as e:
                    log(f"Error reading {smt2_file}: {e}")
    
    id_to_problem = {}
    df_copy["problem"] = None
    
    for index, row in df_copy.iterrows():
        matches = [
            problem.replace(str(spf_path) + "/", "") 
            for problem in problems 
            if row["answer"] in problems[problem]
        ]
        
        if len(matches) == 1:
            df_copy.at[index, "problem"] = matches[0]
            id_to_problem[str(index)] = matches[0]
        elif len(matches) > 1:
            log(f"Ambiguous mapping for row {row['id']}: {matches}")
            id_to_problem[str(index)] = matches[0]
        else:
            log(f"No mapping found for row {row['id']}")
    
    print(f"Created problem mapping for {len(id_to_problem)} entries")
    return id_to_problem


def init_summary() -> Summary:
    return {key: 0 for key in SUMMARY_KEYS}


def find_local_stats_files(root: str = ".") -> list[Path]:
    all_files = []
    search_root = Path(root)
    if search_root.exists():
        found_files = list(search_root.rglob("individual_stats.json"))
        all_files.extend(found_files)
        if found_files:
            log(f"Found {len(found_files)} stats files in {root}")
    return all_files


def extract_local_model_name(path: Path) -> str:
    parts = path.parts
    model_name = "unknown_model"

    for i, part in enumerate(parts):
        if part.startswith("results_"):
            if i + 1 < len(parts):
                model_name = parts[i + 1]
                log(f"Found model name '{model_name}' using results_ pattern")
                return model_name
    raise ValueError


def extract_local_trial_info(path: Path) -> str:
    try:
        for part in path.parts:
            if (len(part) >= 10 and 
                part[4] == '-' and part[7] == '-' and
                part[:4].isdigit() and part[5:7].isdigit() and part[8:10].isdigit()):
                log(f"Found trial id '{part}' using date pattern")
                return part

        date_part = path.parts[-3]
        log(f"Found trial id '{date_part}' using parent directory")
        return date_part
    except (IndexError, ValueError) as e:
        log(f"Failed to extract trial info: {e}")
        return "unknown_trial"


def load_local_stats_file(file_path: Path) -> TierStats:
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

    overall_summary = init_summary()
    for problem in all_problems:
        try:
            category = local_reason_to_category(problem)
            overall_summary[category] += 1
        except ValueError as e:
            print(f"Error in overall summary: {e}")

    tier_summaries["overall"] = overall_summary
    return tier_summaries


def find_openai_result_files(root: str = "results") -> list[Path]:
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
    filename = path.stem

    if "-20" in filename:
        model_name = filename.split("-20")[0]
        return model_name.rstrip("-")
    elif "gpt" in filename.lower() or "claude" in filename.lower():
        parts = filename.split("-")
        if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) == 4:
            return parts[0]
        else:
            return filename

    return filename


def extract_openai_trial_id(path: Path) -> str:
    parts = path.parts
    for part in parts:
        if part.startswith("trial"):
            return part
    return "unknown_trial"


def load_openai_result_file(file_path: Path) -> Dict[str, Any]:
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
    tier_summaries = defaultdict(init_summary)
    all_results = init_summary()

    for result in results:
        try:
            reason = result.get("reason")
            category = openai_reason_to_category(reason)

            custom_id = int(result.get("custom_id", -1))
            tier = tier_mapping.get(custom_id, "unknown_tier")

            tier_summaries[tier][category] += 1
            all_results[category] += 1
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Result causing error: {result}")

    tier_summaries["overall"] = all_results
    return dict(tier_summaries)


def categorize_result(result):
    if not isinstance(result, dict):
        return "unknown"
        
    if "z3_result" in result:
        if result["z3_result"] is True:
            return "correct"
    
    if result.get("result") is True:
        return "correct"
    
    reason = result.get("reason", "")
    if not isinstance(reason, str):
        return "unknown"
        
    if reason.startswith((FORMAT_ERROR, FORMAT_ERROR_2)) or reason == REASON_FORMAT or "Failed to extract" in reason:
        return "error_output_formatting"
    elif reason.startswith(PARSE_ERROR) or reason == REASON_SYNTAX or "Could not parse" in reason:
        return "error_syntax"
    elif "does not imply" in reason or reason in (REASON_SEMANTICS_A, REASON_SEMANTICS_B):
        return "error_semantics"
    elif reason == REASON_CORRECT or reason == "Constraints are logically equivalent.":
        return "correct"
    else:
        return "unknown"


def analyze_models_by_problem(problem_mapping, output_dir="output", verbose=False):
    def log_problem(msg):
        if verbose:
            print(f"[PROBLEM] {msg}")
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    def get_problem_name(problem_id):
        if problem_id is None:
            return "unknown"
        
        problem_id_str = str(problem_id)
        
        if problem_id_str in problem_mapping:
            return problem_mapping[problem_id_str]
        
        try:
            idx = int(problem_id_str)
            if str(idx) in problem_mapping:
                return problem_mapping[str(idx)]
        except (ValueError, TypeError):
            pass

            
        return problem_id_str
    
    local_models = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    openai_models = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    local_files = find_local_stats_files()
    log_problem(f"Processing {len(local_files)} local model result files")
    
    for file_path in local_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            model_name = extract_local_model_name(file_path)
            log_problem(f"Processing {file_path} for model {model_name}")
            
            if isinstance(data, dict) and any(k in data for k in ["small", "medium", "large"]):
                for difficulty, items in data.items():
                    if isinstance(items, list):
                        log_problem(f"Processing {len(items)} items in {difficulty} category")
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                                
                            problem_id = item.get("index")
                            if problem_id is None:
                                continue
                                
                            problem_name = get_problem_name(problem_id)
                            category = categorize_result(item)
                            
                            local_models[model_name][problem_name][category] += 1
                            local_models[model_name][problem_name]["total_attempts"] += 1
            elif isinstance(data, list):
                log_problem(f"List format detected in {file_path}, length: {len(data)}")
                for item in data:
                    if not isinstance(item, dict):
                        continue
                        
                    problem_id = item.get("id") or item.get("index")
                    problem_name = get_problem_name(problem_id)
                    category = categorize_result(item)
                    
                    local_models[model_name][problem_name][category] += 1
                    local_models[model_name][problem_name]["total_attempts"] += 1
            elif isinstance(data, dict):
                log_problem(f"Dictionary format detected in {file_path}, keys: {len(data)}")
                for problem_id, problem_data in data.items():
                    if not isinstance(problem_data, dict):
                        continue
                        
                    actual_problem_id = problem_data.get("id") or problem_data.get("index") or problem_id
                    problem_name = get_problem_name(actual_problem_id)
                    category = categorize_result(problem_data)
                    
                    local_models[model_name][problem_name][category] += 1
                    local_models[model_name][problem_name]["total_attempts"] += 1
                
        except Exception as e:
            log_problem(f"Error processing {file_path}: {str(e)}")
    
    openai_files = find_openai_result_files()
    log_problem(f"Processing {len(openai_files)} OpenAI model result files")
    
    for file_path in openai_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if "results" not in data:
                log_problem(f"No results in {file_path}")
                continue
                
            model_name = extract_openai_model_name(file_path)
            
            for result in data["results"]:
                if not isinstance(result, dict):
                    continue
                    
                problem_id = result.get("custom_id")
                problem_name = get_problem_name(problem_id)
                category = categorize_result(result)
                
                openai_models[model_name][problem_name][category] += 1
                openai_models[model_name][problem_name]["total_attempts"] += 1
                
        except Exception as e:
            log_problem(f"Error processing {file_path}: {str(e)}")
    
    with open(f"{output_dir}/local_model_problem_stats.json", "w") as f:
        json.dump(local_models, f, indent=2)
    
    with open(f"{output_dir}/openai_model_problem_stats.json", "w") as f:
        json.dump(openai_models, f, indent=2)
    
    problem_dir = Path(output_dir) / "problems"
    problem_dir.mkdir(exist_ok=True)
    
    all_problems = set()
    for model_data in local_models.values():
        all_problems.update(model_data.keys())
    for model_data in openai_models.values():
        all_problems.update(model_data.keys())
    
    for problem in all_problems:
        if problem == "unknown":
            continue
            
        problem_report = {
            "problem": problem,
            "local_models": {},
            "openai_models": {}
        }
        
        for model, problems in local_models.items():
            if problem in problems:
                problem_report["local_models"][model] = problems[problem]
                
        for model, problems in openai_models.items():
            if problem in problems:
                problem_report["openai_models"][model] = problems[problem]
        
        safe_name = problem.replace("/", "_").replace(":", "-")
        with open(f"{problem_dir}/{safe_name}.json", "w") as f:
            json.dump(problem_report, f, indent=2)
    
    print(f"Problem-based analysis complete. Reports written to:")
    print(f"- {output_dir}/local_model_problem_stats.json")
    print(f"- {output_dir}/openai_model_problem_stats.json")
    print(f"- {output_dir}/problems/ (per-problem reports)")
    
    return {
        "local_models": dict(local_models),
        "openai_models": dict(openai_models)
    }


def calculate_percentages(summary: Summary) -> dict[str, float]:
    total = sum(summary.values())
    return {key: (value / total) * 100 if total else 0 for key, value in summary.items()}


def write_summary_json(summary_dict: dict[str, Any], path: Path):
    with path.open("w") as f:
        json.dump(summary_dict, f, indent=2)


def process_local_models(tier_mapping, output_dir="output"):
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
            if any(tier in ["small", "medium", "large"] for tier in stats.keys()):
                log(f"File already has tier information")
                tier_summaries = syntax_semantics_summary_by_tier(stats)
            else:
                log(f"Restructuring data with tier information")
                tiered_stats = defaultdict(list)
                for problem_id, problem_data in stats.items():
                    try:
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

    detailed_results = {}
    tier_merged_summaries = defaultdict(lambda: defaultdict(init_summary))

    for model, trials in model_trial_data.items():
        detailed_results[model] = {
            "trials": {},
            "aggregate_by_tier": defaultdict(init_summary),
            "aggregate": init_summary(),
            "trial_count": len(trials),
        }

        log(f"Processing {len(trials)} trials for model {model}")

        for trial_id, trial_data in trials.items():
            tier_data = trial_data["tier_summaries"]

            detailed_results[model]["trials"][trial_id] = {
                "tier_summaries": {},
                "overall": tier_data["overall"],
                "overall_percentages": calculate_percentages(tier_data["overall"]),
            }

            for tier, summary in tier_data.items():
                if tier != "overall":
                    detailed_results[model]["trials"][trial_id]["tier_summaries"][tier] = {
                        "summary": summary,
                        "percentages": calculate_percentages(summary),
                    }

                    for category, count in summary.items():
                        tier_merged_summaries[model][tier][category] += count

        for tier, summary in tier_merged_summaries[model].items():
            detailed_results[model]["aggregate_by_tier"][tier] = summary
            detailed_results[model]["aggregate_by_tier"][f"{tier}_percentages"] = calculate_percentages(summary)

        if "overall" in tier_merged_summaries[model]:
            detailed_results[model]["aggregate"] = tier_merged_summaries[model]["overall"]
        else:
            combined_summary = init_summary()
            for tier, summary in tier_merged_summaries[model].items():
                if tier != "overall" and not tier.endswith("_percentages"):
                    for category, count in summary.items():
                        combined_summary[category] += count

            detailed_results[model]["aggregate"] = combined_summary
            tier_merged_summaries[model]["overall"] = combined_summary

        detailed_results[model]["aggregate_percentages"] = calculate_percentages(detailed_results[model]["aggregate"])

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    tier_summaries = {}
    for model, tiers in tier_merged_summaries.items():
        tier_summaries[model] = {}
        for tier, summary in tiers.items():
            if not tier.endswith("_percentages"):
                tier_summaries[model][tier] = dict(summary)

    write_summary_json(tier_summaries, output_path / "local_model_tier_summary.json")
    write_summary_json(detailed_results, output_path / "local_model_detailed_summary.json")

    overall_summaries = {model: data["aggregate"] for model, data in detailed_results.items()}
    write_summary_json(overall_summaries, output_path / "local_model_summary.json")
    write_summary_json(overall_summaries, output_path / "model_summary.json")

    print("Local model summaries written to:")
    print(f"   - {output_path}/local_model_tier_summary.json (tier-based summaries)")
    print(f"   - {output_path}/local_model_detailed_summary.json (detailed with trials)")
    print(f"   - {output_path}/local_model_summary.json (simple aggregate)")
    print(f"   - {output_path}/model_summary.json (for compatibility)")


def process_openai_models(tier_mapping, output_dir="output"):
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

        tier_summaries = generate_openai_summary_by_tier(data["results"], tier_mapping)
        model_data[model_name][trial_id] = {"tier_summaries": tier_summaries}

    detailed_results = {}
    tier_merged_summaries = defaultdict(lambda: defaultdict(init_summary))

    for model, trials in model_data.items():
        detailed_results[model] = {
            "trials": {},
            "aggregate_by_tier": defaultdict(init_summary),
            "aggregate": init_summary(),
            "trial_count": len(trials),
        }

        log(f"Processing {len(trials)} trials for OpenAI model {model}")

        for trial_id, trial_data in trials.items():
            tier_data = trial_data["tier_summaries"]

            detailed_results[model]["trials"][trial_id] = {
                "tier_summaries": {},
                "overall": tier_data["overall"],
                "overall_percentages": calculate_percentages(tier_data["overall"]),
            }

            for tier, summary in tier_data.items():
                if tier != "overall":
                    detailed_results[model]["trials"][trial_id]["tier_summaries"][tier] = {
                        "summary": summary,
                        "percentages": calculate_percentages(summary),
                    }

                    for category, count in summary.items():
                        tier_merged_summaries[model][tier][category] += count

        for tier, summary in tier_merged_summaries[model].items():
            detailed_results[model]["aggregate_by_tier"][tier] = summary
            detailed_results[model]["aggregate_by_tier"][f"{tier}_percentages"] = calculate_percentages(summary)

        if "overall" in tier_merged_summaries[model]:
            detailed_results[model]["aggregate"] = tier_merged_summaries[model]["overall"]
        else:
            combined_summary = init_summary()
            for tier, summary in tier_merged_summaries[model].items():
                if tier != "overall" and not tier.endswith("_percentages"):
                    for category, count in summary.items():
                        combined_summary[category] += count

            detailed_results[model]["aggregate"] = combined_summary
            tier_merged_summaries[model]["overall"] = combined_summary

        detailed_results[model]["aggregate_percentages"] = calculate_percentages(detailed_results[model]["aggregate"])

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    tier_summaries = {}
    for model, tiers in tier_merged_summaries.items():
        tier_summaries[model] = {}
        for tier, summary in tiers.items():
            if not tier.endswith("_percentages"):
                tier_summaries[model][tier] = dict(summary)

    write_summary_json(tier_summaries, output_path / "openai_model_tier_summary.json")
    write_summary_json(detailed_results, output_path / "openai_model_detailed_summary.json")

    overall_summaries = {model: data["aggregate"] for model, data in detailed_results.items()}
    write_summary_json(overall_summaries, output_path / "openai_model_summary.json")

    existing_summary_file = output_path / "model_summary.json"
    if existing_summary_file.exists():
        try:
            with existing_summary_file.open() as f:
                existing_summaries = json.load(f)
            merged_all = {**existing_summaries, **overall_summaries}
            write_summary_json(merged_all, output_path / "model_summary.json")
            print(f"Merged OpenAI models with existing summaries.")
        except Exception as e:
            print(f"Warning: Failed to merge with existing summaries: {e}")
    else:
        write_summary_json(overall_summaries, output_path / "model_summary.json")

    print("OpenAI model summaries written to:")
    print(f"   - {output_path}/openai_model_tier_summary.json (tier-based summaries)")
    print(f"   - {output_path}/openai_model_detailed_summary.json (detailed with trials)")
    print(f"   - {output_path}/openai_model_summary.json (OpenAI models only)")
    print(f"   - {output_path}/model_summary.json (all models combined)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results with comprehensive analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--local-path", default=".", help="Path to search for local model results (default: current directory)")
    parser.add_argument("--openai-path", default="results", help="Path to search for OpenAI model results (default: results)")
    parser.add_argument("--output-dir", default="output", help="Output directory for results (default: output)")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    print("Starting comprehensive model evaluation analysis...")
    
    benchmark_df = load_benchmark_dataset()
    tier_mapping = create_tier_mapping(benchmark_df)

    print(f"\nFound tier information for {len(tier_mapping)} problems")
    tier_counts = defaultdict(int)
    for tier in tier_mapping.values():
        tier_counts[tier] += 1

    for tier, count in tier_counts.items():
        print(f"  - {tier}: {count} problems")

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    global find_local_stats_files, find_openai_result_files
    original_find_local = find_local_stats_files
    original_find_openai = find_openai_result_files
    
    find_local_stats_files = lambda: list(Path(args.local_path).rglob("individual_stats.json")) if Path(args.local_path).exists() else []
    find_openai_result_files = lambda: [f for trial_dir in Path(args.openai_path).glob("trial*") if trial_dir.is_dir() for f in trial_dir.glob("*.json") if "summary" not in f.name] if Path(args.openai_path).exists() else []

    try:
        local_files = find_local_stats_files()
        if local_files:
            print(f"\nProcessing {len(local_files)} local model files from: {args.local_path}")
            process_local_models(tier_mapping, args.output_dir)
        else:
            print(f"\nNo local model files found in: {args.local_path}")

        openai_files = find_openai_result_files()
        if openai_files:
            print(f"\nProcessing {len(openai_files)} OpenAI model files from: {args.openai_path}")
            process_openai_models(tier_mapping, args.output_dir)
        else:
            print(f"\nNo OpenAI model files found in: {args.openai_path}")

        print("\nPerforming problem-based analysis...")
        problem_mapping = create_problem_mapping(benchmark_df, "spf-wca/custom")
        if problem_mapping:
            analyze_models_by_problem(problem_mapping, output_dir=args.output_dir, verbose=args.verbose)
        else:
            print("Skipping problem-based analysis due to missing problem mapping.")

    finally:
        find_local_stats_files = original_find_local
        find_openai_result_files = original_find_openai

    print(f"\nAll processing complete. Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()