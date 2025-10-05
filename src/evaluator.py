import argparse
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.configs import EvaluationConfig, ModelConfig, load_config
from utils.evaluation import (
    EvaluationStats,
    LLMHelper,
    Loader,
    check_logical_equivalence,
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
        evalcfg: EvaluationConfig,
    ) -> None:
        self.model = LLMHelper(modelconfig=modelcfg)
        self.configs = evalcfg
        self.configs.set_dirs()
        self.stats = EvaluationStats()

        # Load HF dataset by identifier
        ds = load_dataset(self.configs.dataset, split="test")

        ds = ds.map(
            lambda _, idx: {"index": idx},
            with_indices=True,
            remove_columns=[],
        )

        # 1. Precompute prompts:
        def make_prompt(example):
            example["question"] = Loader.apply_chat_template(
                prompt=example["question"],
                instruct=modelcfg.instruct
            )
            return example


        ds = ds.map(
            make_prompt,
            batched=False,
            batch_size=evalcfg.batch_size,
            num_proc=8,
            remove_columns=[])
        # 2. Tell HF to return PyTorch tensors for the columns you need:
        ds.set_format("torch", columns=["index","tier","question","answer","constants"])

        self.dataloader = DataLoader(
            ds,
            batch_size=self.configs.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )

    def run(self) -> None:
        for batch in self.dataloader:
            indices:    list[int]              = batch["index"].tolist()
            problems:   list[str]              = batch["tier"]
            questions:  list[str]              = batch["question"]
            truths:     list[str]              = batch["answer"]
            constants:  list[str | None]    = batch.get("constants", [None]*len(indices))

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

    cfg = load_config()
    print("=== Loaded Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    modelcfg = ModelConfig(**cfg.model)
    evalcfg = EvaluationConfig(**cfg.evaluation)



    exp = Experiment(
        modelcfg=modelcfg,
        evalcfg=evalcfg,
    )
    exp.run()
