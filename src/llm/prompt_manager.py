from core.settings import settings
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptManager:
    def __init__(self, prompt_dir: str = settings.prompt_dir):
        self._env = Environment(
            loader=FileSystemLoader(prompt_dir),
            undefined=StrictUndefined,
        )

    def get_prompt(self, prompt_name: str):
        template_path = f"{prompt_name}.j2"
        template = self._env.get_template(template_path)
        return template
