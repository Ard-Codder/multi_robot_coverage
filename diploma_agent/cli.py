"""CLI entry point for the local thesis agent."""

from __future__ import annotations

import argparse

from diploma_agent.orchestrator import DiplomaOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Local thesis agent commands")
    parser.add_argument("command", nargs="*", help="Command, for example: /write 2.1")
    args = parser.parse_args()
    command = " ".join(args.command).strip() or "/plan"
    result = DiplomaOrchestrator().handle(command)
    print(result.message)
    if result.content:
        print()
        print(result.content)


if __name__ == "__main__":
    main()
