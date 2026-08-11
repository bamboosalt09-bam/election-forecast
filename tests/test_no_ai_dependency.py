import ast
from pathlib import Path


def test_presidential_modules_do_not_import_ai_or_news_packages() -> None:
    blocked_roots = {"openai", "anthropic", "sklearn", "tensorflow", "torch", "news_collector", "news_analyzer"}
    for path in Path("src/election_forecast/presidential").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert not names & blocked_roots
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in blocked_roots

