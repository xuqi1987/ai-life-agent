"""CLI entry point."""

import sys


def main():
    """Run the agent from command line."""
    print("minimax-agent v0.1.0 — MiniMax M2.7 Multi-modal Agent")
    print("Type 'quit' or 'exit' to stop.\n")

    from minimax_agent.core.agent import Agent

    agent = Agent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = agent.run(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
