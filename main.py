"""Command-line entry point for the simplified coding agent."""

import argparse
from pathlib import Path

from agent import CodingAgent
from config.settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Run a simplified coding agent.")
    parser.add_argument("task", help="Task for the agent to work on.")
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Optional JSONL history path.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum tool-calling loop iterations.",
    )
    return parser


def main() -> None:
    """Parse arguments, run the agent, and print the final answer."""

    args = build_parser().parse_args()
    settings = get_settings()
    agent = CodingAgent(
        settings=settings,
        history_path=args.history,
        max_iterations=args.max_iterations,
    )
    print(agent.run(args.task))


if __name__ == "__main__":
    main()
