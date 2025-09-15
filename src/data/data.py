import argparse
import re
from pathlib import Path

import pandas as pd


class PromptBuilder:
    """
    A class for building prompt questions.

    The prompt template can be customized. By default, if instruct mode is enabled,
    the prompt will be built in an instruct style; otherwise, in a plain style.
    The prompt style follows the qwen2.5 format.

    The template uses the following placeholders:
      - {examples}: formatted examples with their N values and constraints
      - {question}: the target N value to generate constraints for
    """

    def __init__(self, instruct: bool, template: str = None) -> None:
        self.instruct = instruct
        if template:
            self.template = template
        elif instruct:
            self.template = """<|im_start|>system
You are a helpful assistant.
You first think about the reasoning process in your mind and then provide the user with the answer.
<|im_end|>
<|im_start|>user
Your role is to take a known pattern of symbolic constraints that represent the longest execution path of a program
and generalize it for any given input size N.
All per-variable constraints must be combined using a top-level (assert (and ...)) clause.
Show your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.
For example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.
Here are the known constraints:
{examples}
What is the constraint for N={question}?
<|im_end|>
"""
        else:
            self.template = """You are a helpful assistant.
You first think about the reasoning process in your mind and then provide the user with the answer.
User: Your role is to take a known pattern of symbolic constraints that represent the longest execution path of a program
and generalize it for any given input size N.
All per-variable constraints must be combined using a top-level (assert (and ...)) clause.
Show your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.
For example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.
Here are the known constraints:
{examples}
What is the constraint for N={question}?
"""

    def build_prompt(self, examples: str, question: str) -> str:
        """
        Build a prompt using the template.

        Args:
            examples: The formatted examples with their N values and constraints.
            question: The target N value to generate constraints for.

        Returns:
            A formatted prompt string.
        """
        return self.template.format(examples=examples, question=question)


def discover(folder: Path, prefix: str = "custom") -> dict[str, dict[int, dict[str, list[str]]]]:
    """
    Discover SMT2 files from a folder and its subfolders.

    Expected filename format: {prefix}.{problem}_{number}.smt2
    e.g., custom.ProblemName_1.smt2

    Returns:
        A nested dictionary mapping each problem to its example numbers.
        Each example maps to a dict with keys "constants" and "assertions".
    """
    discovered = {}
    pattern_re = re.compile(rf"^{re.escape(prefix)}\.([^.]+)_([0-9]+)\.smt2$")
    smt_files = folder.rglob(f"{prefix}.*_*.smt2")
    for smt_file in smt_files:
        filename = smt_file.name
        match = pattern_re.match(filename)
        if not match:
            continue
        problem, n_str = match.groups()
        try:
            n = int(n_str)
        except ValueError:
            continue
        with smt_file.open() as file:
            smt_lines = [line.strip() for line in file if line.strip()]
            constants = [line for line in smt_lines if line.startswith("(declare-const")]
            assertions = [line for line in smt_lines if line.startswith("(assert")]
        if problem not in discovered:
            discovered[problem] = {}
        discovered[problem][n] = {"constants": constants, "assertions": assertions}
    print(f"Discovered {len(discovered)} problems with {sum(len(v) for v in discovered.values())} examples")
    return discovered


def build_dataset(
    discovered: dict[str, dict[int, dict[str, list[str]]]],
    min_n_value: int,
    max_n_value: int,
    prompt_builder: PromptBuilder,
    max_target_n: int = 30,
    min_example_count: int = 2,
    max_example_count: int = 5,
    samples_per_target: int = 3,
    sep: str = "\n",
    random_seed: int = 42069,
) -> pd.DataFrame:
    """
    Build a dataset DataFrame from discovered SMT2 files with comprehensive examples.

    For each problem and each valid target N:
      - Multiple sets of examples are created
      - Examples are ordered to show progression pattern

    Args:
        discovered: Dictionary mapping problems to their examples.
        min_n_value: Minimum N value to include from files.
        max_n_value: Maximum N value to include from files.
        prompt_builder: An instance of PromptBuilder to generate questions.
        max_target_n: Maximum value for target N (default 30).
        min_example_count: Minimum number of examples per prompt (default 2).
        max_example_count: Maximum number of examples per prompt (default 5).
        samples_per_target: Number of different example sets to create per target N.
        sep: Separator for joining multiple lines (default newline).
        random_seed: Seed for reproducibility.

    Returns:
        A pandas DataFrame with columns: problem, question, question_index, answer, constants.
    """
    import random

    random.seed(random_seed)

    rows = []
    for problem, examples in discovered.items():
        # Get available example numbers within the specified range
        available_examples = [n for n in examples.keys() if min_n_value <= n <= max_n_value]

        # Skip if not enough examples
        if len(available_examples) < min_example_count:
            continue

        # Find potential target values (values we have ground truth for)
        potential_targets = [n for n in examples.keys() if n > min(available_examples) and n <= max_target_n]

        # Skip if no valid targets
        if not potential_targets:
            continue

        # For each target N, create multiple example sets
        for target_n in potential_targets:
            # Select available examples below the target
            valid_examples = [n for n in available_examples if n < target_n]

            # Skip if not enough valid examples
            if len(valid_examples) < min_example_count:
                continue

            # Create multiple sets of examples for this target
            for _ in range(samples_per_target):
                # Choose a random number of examples
                num_examples = random.randint(min_example_count, min(max_example_count, len(valid_examples)))

                # Randomly select examples and sort them (to maintain order)
                selected_examples = sorted(random.sample(valid_examples, num_examples))

                # Format examples as "N=1: (constraints)", "N=2: (constraints)", etc.
                formatted_examples = []
                for ex_num in selected_examples:
                    data = examples[ex_num]
                    example_text = sep.join(data.get("assertions", []))
                    formatted_examples.append(f"N={ex_num}: {example_text}")

                # Join all formatted examples
                examples_text = sep + sep.join(formatted_examples)

                # Build prompt asking for the target N
                question = prompt_builder.build_prompt(examples=examples_text, question=str(target_n))

                # Get the ground truth answer
                target_data = examples.get(target_n, {})
                answer = sep.join(target_data.get("assertions", []))
                constants = sep.join(target_data.get("constants", []))

                rows.append(
                    {
                        "problem": problem,
                        "question": question,
                        "question_index": target_n,
                        "answer": answer,
                        "constants": constants,
                        "examples_used": selected_examples,
                    }
                )

    print(f"Created {len(rows)} dataset rows across {len(discovered)} problems")
    return pd.DataFrame(rows)


def main() -> None:
    """
    Discover SMT2 files, build a dataset with prompts, answers, and constants,
    and save it as a parquet file.

    The question template can be customized via command-line arguments,
    and instruct mode is selectable.
    """
    parser = argparse.ArgumentParser(
        description="Build a dataset with columns: question, answer, constants. "
        "The question is built using a customizable prompt template."
    )
    parser.add_argument("folder", type=str, help="Parent folder to search for SMT2 files")
    parser.add_argument("--prefix", type=str, default="custom", help="Prefix used in SMT2 filenames")
    parser.add_argument("--max_target_n", type=int, default=15, help="Maximum value for target N (default 30)")
    parser.add_argument(
        "--min_example_count",
        type=int,
        default=2,
        help="Minimum number of examples to include in each prompt (default 2)",
    )
    parser.add_argument(
        "--max_example_count",
        type=int,
        default=5,
        help="Maximum number of examples to include in each prompt (default 5)",
    )
    parser.add_argument("--min_n_value", type=int, default=1, help="Minimum N value to include from files (default 1)")
    parser.add_argument(
        "--max_n_value", type=int, default=15, help="Maximum N value to include from files (default 30)"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42069, help="Random seed for reproducibility (default 42069)"
    )
    parser.add_argument("--instruct", action="store_true", help="Use instruct-style prompt")
    parser.add_argument("--template", type=str, default=None, help="Custom question template (optional)")
    parser.add_argument("--separator", type=str, default="\n", help="Separator for joining multiple lines")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the parquet file")
    parser.add_argument(
        "--output_filename", type=str, default="dataset.parquet", help="Name of the output parquet file"
    )
    parser.add_argument(
        "--samples_per_target",
        type=int,
        default=5,
        help="Number of different example sets to create per target N (default 5)",
    )
    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of rows to sample (default 1.0)")
    args = parser.parse_args()

    folder = Path(args.folder)
    discovered = discover(folder, prefix=args.prefix)
    prompt_builder = PromptBuilder(instruct=args.instruct, template=args.template)
    df = build_dataset(
        discovered,
        min_n_value=args.min_n_value,
        max_n_value=args.max_n_value,
        prompt_builder=prompt_builder,
        max_target_n=args.max_target_n,
        min_example_count=args.min_example_count,
        max_example_count=args.max_example_count,
        samples_per_target=args.samples_per_target,
        sep=args.separator,
        random_seed=args.random_seed,
    )
    df = df.groupby("problem").apply(lambda x: x.sample(frac=args.frac,random_state=69420)).reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename
    df.to_parquet(output_path, index=False)
    print(f"Saved dataset with {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
