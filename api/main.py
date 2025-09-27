import logging
from api.chatbot_inference import Chatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_chatbot():
    chatbot = Chatbot()
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        result = chatbot.generate_response(user_input)
        print(f"Bot: {result['response']} (Intent: {result['intent']} [{result['intent_confidence']:.2f}], "
              f"Response Confidence: {result['response_confidence']:.2f})")

if __name__ == "__main__":
    run_chatbot()
