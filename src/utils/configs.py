"""
Configuration classes for the different components of the project.
"""

from pathlib import Path
from datetime import datetime


class EvaluationConfig:
    """
    Configuration for experiment/evaluator.py.
    """

    def __init__(
        self,
        results_dir,
        attempts=1,
        max_file_size=2_000_000,
    ) -> None:
        """
        Initialize the evaluation configuration.

        Args:
            results_dir (str): The directory to save the results
            attempts (int): The number of attempts for the evaluation.
            max_file_size (int): The maximum file size when crafting .smt2.

        """
        self.max_file_size = max_file_size
        results_dir = Path(results_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logs_dir = Path(results_dir) / "logs"
        self.stats_dir = Path(results_dir) / "stats"
        self.generals = Path(results_dir) / "generals"
        self.attempts = attempts

    def set_dirs(self) -> None:
        """
        Set the directories for the evaluation configuration.

        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.generals.mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """
    Configuration for the model.
    """

    def __init__(
        self,
        model_name: str,
        quantization_mode: str | None = None,
        token: str | None = None,
        instruct: bool | None = None,
    ) -> None:
        """
        Initialize the VLLM Helper.

        Args:
            model_name: The model to use.
            think: Whether think mode is enabled.
            quantization_mode: The quantization mode.
            token: The Hugging Face token.

        """
        self.openai = bool(model_name.startswith("openai"))
        self.model = model_name.removeprefix("openai/")
        self.quantization = quantization_mode
        self.token = token
        self.is_instruct = instruct
