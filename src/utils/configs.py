"""
Configuration classes for the different components of the project.

- SamplingParamsConfig: Configuration for all sampling parameters used in text generation (mirrors vLLM SamplingParams).
- ModelConfig: Configuration for the model and its runtime options.
- EvaluationConfig: Configuration for experiment/evaluator.py output and run settings.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config() -> "OmegaConf.DictConfig":
    """
    Load a base YAML config and apply any dotlist-style CLI overrides.

    Example:
        python src/evaluator.py model.model_name=meta/llama3 evaluation.batch_size=64.

    """
    parser = argparse.ArgumentParser(description="Evaluation configuration loader.")
    # Collect *everything* after known args (e.g. model.model_name=foo)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base_cfg = OmegaConf.load(Path(__file__).parent / "configs.yaml")

    if args.overrides:
        override_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg

    return cfg


@dataclass
class SamplingParamsConfig:
    """
    Configuration for all sampling parameters used in text generation.

    Attributes:
        n (int): Number of outputs to return for the given prompt request.
        presence_penalty (float): Penalizes new tokens based on presence.
        frequency_penalty (float): Penalizes new tokens based on frequency.
        repetition_penalty (float): Penalizes new tokens based on repetition.
        temperature (float): Controls randomness of sampling.
        top_p (float): Cumulative probability of top tokens to consider.
        top_k (int): Number of top tokens to consider.
        min_p (float): Minimum probability for a token to be considered.
        seed (Optional[int]): Random seed for generation.
        stop (Optional[Union[str, list[str]]]): String(s) that stop generation.
        stop_token_ids (Optional[list[int]]): Token IDs that stop generation.
        bad_words (Optional[list[str]]): Words not allowed to be generated.
        include_stop_str_in_output (bool): Include stop strings in output text.
        ignore_eos (bool): Ignore EOS token and continue generating.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        min_tokens (int): Minimum number of tokens to generate.
        logprobs (Optional[int]): Number of log probabilities to return per output token.
        prompt_logprobs (Optional[int]): Number of log probabilities to return per prompt token.
        skip_special_tokens (bool): Whether to skip special tokens in output.
        spaces_between_special_tokens (bool): Add spaces between special tokens.
        truncate_prompt_tokens (Optional[int]): Prompt truncation size.
        extra_args (Optional[dict]): Arbitrary additional args.

    """

    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    max_tokens: int | None = 16
    min_tokens: int = 0
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: int | None = None
    extra_args: dict | None = None


@dataclass
class ModelConfig:
    """
    Configuration for the model and its runtime options.

    Attributes:
        model_name (str): The model to use.
        quantization_mode (Optional[str]): The quantization mode.
        token (Optional[str]): The Hugging Face token.
        instruct (Optional[bool]): Whether instruct mode is enabled.
        pipeline_parallelism (int): Number of pipeline-parallel workers.

    """

    model_name: str
    quantization_mode: str | None = None
    instruct: bool | None = None
    pipeline_parallelism: int = 1
    sampling: SamplingParamsConfig = None

    def __post_init__(self):
        self.openai = bool(self.model_name.startswith("openai"))
        self.model = self.model_name.removeprefix("openai/")
        self.sampling = SamplingParamsConfig(**self.sampling)

@dataclass
class EvaluationConfig:
    """
    Configuration for experiment/evaluator.py output and run settings.

    Attributes:
        results_dir (str): The directory to save the results.
        attempts (int): The number of attempts for the evaluation.
        max_file_size (int): The maximum file size when crafting .smt2.
        logs_dir (Path): Directory for logs.
        stats_dir (Path): Directory for stats.
        generals (Path): Directory for general outputs.

    """

    results_dir: str
    attempts: int = 1
    max_file_size: int = 2_000_000
    batch_size: int | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        """
        Initialize output directories based on the current timestamp.
        """
        results_dir = Path(self.results_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logs_dir = results_dir / "logs"
        self.stats_dir = results_dir / "stats"
        self.generals = results_dir / "generals"

    def set_dirs(self) -> None:
        """
        Create output directories if they do not already exist.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.generals.mkdir(parents=True, exist_ok=True)
