import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import openai
import pandas as pd
import torch
from tqdm import tqdm

from utils.configs import ModelConfig


def check_logical_equivalence(
    original_assertions: str,
    generated_assertions: str,
    constants: Optional[str] = None
) -> Dict[str, Any]:
    """
    In-process Z3 two-step equivalence check:
      1) A ⇒ B
      2) B ⇒ A.
    """
    from z3 import Solver, parse_smt2_string, Not, And, Z3Exception, sat

    orig = original_assertions.strip()
    gen = generated_assertions.strip()
    if constants:
        decls = constants.strip()
        orig = decls + f"\n{orig}"
        gen = decls + f"\n{gen}"

    # trivial cases
    if not orig and not gen:
        return {"result": True}
    if not orig or not gen:
        return {"result": False, "reason": "Empty side"}

    try:
        A = parse_smt2_string(orig)
        B = parse_smt2_string(gen)
    except Z3Exception as e:
        return {"result": False, "reason": f"Parse error: {e}"}

    s = Solver()
    # A ⇒ B
    s.push()
    s.add(*A)
    s.add(Not(And(*B)))
    if s.check() == sat:
        return {"result": False, "reason": "A does not imply B"}
    s.pop()
    # B ⇒ A
    s.push()
    s.add(*B)
    s.add(Not(And(*A)))
    if s.check() == sat:
        return {"result": False, "reason": "B does not imply A"}
    s.pop()

    return {"result": True}


class EvaluationStats:
    """
    A class to collect and aggregate evaluation results.
    """
    def __init__(self) -> None:
        self.data: Dict[str, list] = {}
        self.overall: Dict[str, Any] = {}

    def save(self, path: str) -> None:
        """
        Save per-item and overall stats to JSON files.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "individual_stats.json", "w") as f:
            json.dump(self.data, f, indent=2)
        with open(p / "overall_stats.json", "w") as f:
            json.dump(self.overall, f, indent=2)

    def calculate_results(self) -> None:
        """
        Compute overall accuracy from self.data entries.
        """
        total = 0
        correct = 0
        for _, records in self.data.items():
            for rec in records:
                total += 1
                if rec.get("result"):
                    correct += 1
        self.overall = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
        }


class Loader:
    """
    Data loading and response extraction utilities.
    """

    @staticmethod
    def load_data(file: str) -> pd.DataFrame:
        """
        Load and sort a parquet file into a DataFrame.
        """
        return (
            pd.read_parquet(file)
        )

    @staticmethod
    def extract_response(response: str) -> str:
        """
        Parse out the content between <answer> tags.
        """
        try:
            return response.split("<answer>")[1].split("</answer>")[0].strip()
        except Exception:
            raise ValueError("Failed to extract <answer> from response.")

    def apply_chat_template(prompt:str, instruct:bool | None = False) -> str:
        """
        Apply a chat template to the prompt.
        """
        instructions = ("All per-variable constraints must be combined using a top-level (assert (and ...)) clause.\nThe output must be in exact, canonical SMT-LIB format without extra commentary in the constraint string.\nShow your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.\nFor example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.")
        if instruct:
            return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instructions}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"You are a helpful assistant.\nUser:\n{instructions}\n{prompt}\nAssistant:\n"

class BaseLLMHelper(ABC):
    """
    Abstract helper for LLM inference.
    """
    def __init__(self, modelconfig: ModelConfig) -> None:
        self.config = modelconfig

    @abstractmethod
    def _get_responses(self, prompts: list[str]) -> str:
        pass


class _VLLMHelper(BaseLLMHelper):
    """
    Use vLLM for batched inference.
    """
    def __init__(self, modelconfig: ModelConfig) -> None:
        from vllm import LLM, SamplingParams

        os.environ["HF_TOKEN"] = modelconfig.token

        self.pipeline = LLM(
            model=modelconfig.model,
            trust_remote_code=True,
            tensor_parallel_size=(torch.cuda.device_count() or 1),
            dtype="auto",
            disable_custom_all_reduce=True,
            # distributed_executor_backend="ray",
            max_num_batched_tokens = 8000,
            max_model_len = 8000
        )
        self.sampling_params = SamplingParams(max_tokens=8000)

    def _get_responses(self, prompts: list[str]) -> list[str]:
        # generate all prompts in one shot
        outputs = self.pipeline.generate(prompts=prompts, sampling_params=self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

class LLMHelper:
    """
    A class to handle the LLM model.
    """

    def __init__(self, modelconfig: ModelConfig) -> None:
        """
        Initialize the LLM Helper.

        Args:
            modelconfig: The model configuration

        """
        self.modelconfig = modelconfig
        self.__setup()

    def __setup(self) -> None:
        """
        Choose the appropriate helper based on the model name.
        """
        if self.modelconfig.openai:
            raise NotImplementedError("OpenAI API is not supported in this version.")
        else:
            self.helper = _VLLMHelper(modelconfig=self.modelconfig)

    def get_response(self, history: list[str]) -> str:
        """
        Get the response from the model.
        """
        return self.helper._get_responses(history)  # noqa: SLF001
