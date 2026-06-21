from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from llm.adapter import LLMAdapter
from rich.console import Console

console = Console()


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    data: dict
    reasoning: str = ""
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseAgent(ABC):

    def __init__(self, name: str, model: str = "fast"):
        self.name = name
        self.llm = LLMAdapter(model=model)
        self.memory: list[dict] = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def run(self, input_data: dict) -> AgentResult:
        pass

    def think(self, user_message: str) -> str:
        self._log(f"Thinking about: {user_message[:80]}...")
        return self.llm.chat(
            system_prompt=self.system_prompt,
            user_message=user_message,
        )

    def think_with_memory(self, user_message: str) -> str:
        if not self.memory:
            self.memory.append({"role": "system", "content": self.system_prompt})

        self.memory.append({"role": "user", "content": user_message})
        response = self.llm.chat_with_history(self.memory)
        self.memory.append({"role": "assistant", "content": response})
        return response

    def reset_memory(self):
        self.memory = []

    def _log(self, message: str):
        console.print(f"[bold cyan][{self.name}][/bold cyan] {message}")

    def _success(self, data: dict, reasoning: str = "") -> AgentResult:
        self._log(f"[green]Done.[/green]")
        return AgentResult(agent_name=self.name, success=True, data=data, reasoning=reasoning)

    def _failure(self, error: str) -> AgentResult:
        self._log(f"[red]Failed: {error}[/red]")
        return AgentResult(agent_name=self.name, success=False, data={}, errors=[error])
