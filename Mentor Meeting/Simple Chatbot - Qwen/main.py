"""Terminal entry point for the simple chatbot."""

from chat_service import ChatService
from config import MAX_HISTORY_TURNS
from history import ChatHistory


def main() -> None:
    """Run the interactive chat loop."""
    service = ChatService(history=ChatHistory(MAX_HISTORY_TURNS))
    print("Simple chatbot. Type 'quit' or 'exit' to leave, 'clear' to reset.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            service.history.clear()
            print("History cleared.")
            continue

        print(f"Assistant: {service.get_reply(user_input)}")


if __name__ == "__main__":
    main()
