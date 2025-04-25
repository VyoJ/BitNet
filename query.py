import requests
import json


def query_model(messages):
    url = "http://localhost:8000/chat"

    payload = {"messages": messages, "max_new_tokens": 100, "temperature": 0.7}

    response = requests.post(url, json=payload)
    return response.json()["response"]


# Example usage
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How are you?"},
    ]

    response = query_model(messages)
    print(f"Assistant: {response}")

    # Continue the conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Tell me about quantum computing."})

    response = query_model(messages)
    print(f"Assistant: {response}")
