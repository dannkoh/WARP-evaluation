import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from utils.configs import ModelConfig, EvaluationConfig
from utils.evaluation import (
    EvaluationStats,
    Loader,
    check_logical_equivalence,
    LLMHelper,
)

# Load environment variables from .env file
load_dotenv()


class Experiment:
    """
    One-shot, batched evaluation over a HuggingFace dataset using Z3,
    with a formatting instruction for the LLM.
    """
    def __init__(
        self,
        modelcfg: ModelConfig,
        dataset_id: str,
        evalcfg: EvaluationConfig,
        batch_size: int = 64,
    ) -> None:
        self.model = LLMHelper(modelconfig=modelcfg)
        self.configs = evalcfg
        self.configs.set_dirs()
        self.stats = EvaluationStats()

        # Load HF dataset by identifier
        ds = load_dataset(dataset_id, split="test")

        ds = ds.map(
            lambda example, idx: {"index": idx},
            with_indices=True,
            remove_columns=[],
        )

        # 1. Precompute prompts:
        def make_prompt(example):
            example["question"] = Loader.apply_chat_template(
                prompt=example["question"],
                instruct=modelcfg.is_instruct
            )
            return example


        ds = ds.map(
            make_prompt,
            batched=False,
            batch_size=64,
            num_proc=8,
            remove_columns=[])
        # 2. Tell HF to return PyTorch tensors for the columns you need:
        ds.set_format("torch", columns=["index","tier","question","answer","constants"])

        self.dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )

    def run(self) -> None:
        for batch in self.dataloader:
            indices:    List[int]              = batch["index"].tolist()
            problems:   List[str]              = batch["tier"]
            questions:  List[str]              = batch["question"]
            truths:     List[str]              = batch["answer"]
            constants:  List[Optional[str]]    = batch.get("constants", [None]*len(indices))

            responses = self.model.get_response(questions)

            for idx, problem, prompt, response, truth, const in zip(
                indices, problems, questions, responses, truths, constants, strict=False):
                extracted = None
                try:
                    extracted = Loader.extract_response(response)
                    result = check_logical_equivalence(
                        original_assertions=truth,
                        generated_assertions=extracted,
                        constants=const,
                    )
                except Exception as e:
                    result = {"result": False, "reason": str(e)}

                # now record idx alongside everything else
                self.stats.data.setdefault(problem, []).append({
                    "index":   idx,
                    "prompt":  prompt,
                    "response":response,
                    "extracted":extracted,
                    "result":  result.get("result", False),
                    "reason":  result.get("reason"),
                })

        # Finalize and save
        self.stats.calculate_results()
        self.stats.save(path=Path(self.configs.stats_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched Z3-based evaluation")
    parser.add_argument(
        "dataset", help="HuggingFace dataset ID",
    )
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Number of examples per LLM batch",
    )
    parser.add_argument(
        "--instruct", action="store_true", help="Enable instruct-style prompts",
    )
    args = parser.parse_args()

    modelcfg = ModelConfig(
        model_name=args.model,
        quantization_mode=None,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        instruct=args.instruct,
    )
    evalcfg = EvaluationConfig(results_dir=f"./results_{args.model}")

    exp = Experiment(
        modelcfg=modelcfg,
        dataset_id=args.dataset,
        evalcfg=evalcfg,
        batch_size=args.batch_size,
    )
    exp.run()
