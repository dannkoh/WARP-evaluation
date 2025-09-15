from pathlib import Path
from datetime import datetime


class ConversationHandler:
    """
    A class to handle printing and saving conversation messages.
    """

    def __init__(self) -> None:
        """
        Initialize the conversation handler.
        """
        self.conversation: list[str] = []
        self.temp: list[str] = []

    def print_and_save(self, new_message: str) -> None:
        """
        Print and save a new message in the conversation.

        Args:
            new_message (str): The message to print and save.

        """
        if self.temp:
            self.push()

        separator = "#" * 60
        print(f"\n{separator}",flush=True)
        print(new_message, flush=True)

        self.conversation.extend([f"\n{separator}\n", new_message])

    def load(self, message: str) -> None:
        """
        Add a message to the conversation into the conversation list.

        Args:
            message (str): The message to add to the conversation.

        """
        formatted_message = f"\n{message}\n"
        print(formatted_message, flush=True)
        self.temp.append(formatted_message)

    def push(self) -> None:
        """
        Add the temporary conversation to the main conversation.
        """
        if self.temp:
            self.conversation.extend(self.temp)
            self.temp.clear()

    def save(self, path: str) -> None:
        """
        Save log to path.
        """
        filepath = Path(path) / f"{datetime.now().strftime('%Y-%m-%d--%H:%M:%S')}.log"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as f:
            f.write("\n".join(self.conversation_history))

    @property
    def conversation_history(self) -> list[str]:
        """
        Return the conversation history.
        """
        self.push()  # Ensure temp messages are included
        return self.conversation
