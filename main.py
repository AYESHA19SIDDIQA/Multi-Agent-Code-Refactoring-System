import argparse
import os
from urllib.parse import urlparse

from agents.parser3 import analyze_repo
from agents.critic import run_critic
from agents.refactor import run_refactor
from agents.tester import run_tester
from agents.reviewer import run_reviewer


def repo_name_from_url(repo_url):
    # Same rule parser3.analyze_repo uses, so the two agree on the output filename.
    return os.path.splitext(os.path.basename(urlparse(repo_url).path))[0]


def run_pipeline(repo_url, model_tier=None, clone_path="repo-clone"):
    repo_name = repo_name_from_url(repo_url)
    parsed_file = os.path.join("parsed files", f"{repo_name}.json")
    report_file = os.path.join("parsed files", f"{repo_name}-report.md")

    print(f"=== 1/4 Parsing {repo_url} ===")
    analyze_repo(repo_url)

    print("\n=== 2/4 Critiquing flagged entities ===")
    run_critic(parsed_file, model_tier=model_tier)

    print("\n=== 3/4 Refactoring critiqued entities ===")
    run_refactor(parsed_file, model_tier=model_tier)

    print("\n=== 4/4 Testing refactors, then final review ===")
    run_tester(parsed_file, repo_path=clone_path)
    run_reviewer(parsed_file, report_file=report_file, model_tier=model_tier)

    print(f"\nDone.\n  Full results: {parsed_file}\n  Report:       {report_file}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run the parser -> critic -> refactor -> tester -> reviewer pipeline on a git repo."
    )
    arg_parser.add_argument(
        "repo_url",
        nargs="?",
        default="https://github.com/Kalebu/Python-Speech-Recognition-.git",
        help="Git URL of the repo to refactor.",
    )
    arg_parser.add_argument(
        "--model-tier",
        choices=["fast", "balanced", "quality", "frontier"],
        default=None,
        help="Local model size/quality tier (see agents/model_config.py). Defaults to MODEL_TIER env var or 'balanced'.",
    )
    args = arg_parser.parse_args()
    run_pipeline(args.repo_url, model_tier=args.model_tier)
