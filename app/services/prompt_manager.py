from pathlib import Path

class PromptManager:
    def __init__(self):
        # Multiple safe locations for prompts
        self.search_paths = [
            Path(__file__).parent.parent / "prompts",   # app/prompts
            Path.cwd() / "app" / "prompts",             # project_root/app/prompts
            Path.cwd() / "prompts"                      # project_root/prompts (optional)
        ]

    def load_prompt(self, name: str) -> str:
        filename = f"{name}.txt"

        for base in self.search_paths:
            file_path = base / filename
            if file_path.exists():
                return file_path.read_text(encoding="utf-8")

        # If not found in ANY path → clear useful error
        raise FileNotFoundError(
            f"Prompt file '{filename}' not found in paths: "
            f"{[str(p) for p in self.search_paths]}"
        )
