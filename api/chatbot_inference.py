import logging
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extract_data import preprocess_and_vectorize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Chatbot:
    def __init__(self):
        try:
            self.intent_model = joblib.load('models/intent_model.pkl')
            self.intent_vectorizer = joblib.load('models/intent_vectorizer.pkl')
            self.intent_encoder = joblib.load('models/intent_encoder.pkl')
            self.response_vectorizer = joblib.load('models/response_vectorizer.pkl')
            self.response_contexts = joblib.load('models/response_contexts.pkl')
            self.response_texts = joblib.load('models/response_texts.pkl')
            logging.info("All models loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Model file missing: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

        # Response templates for intent-specific responses
        self.response_templates = {
            'default': "I'm here to help. What can I do for you?",
            'greeting': "Hello! How can I assist you today?",
            'book_flight': "Sure, I can help book a flight. Where to?",
            'check_balance': "Checking your balance... (simulated)",
        }

    def classify_intent(self, text):
        try:
            X = preprocess_and_vectorize([text], vectorizer=self.intent_vectorizer)
            pred = self.intent_model.predict(X)[0]
            probs = self.intent_model.predict_proba(X)[0]
            intent = self.intent_encoder.inverse_transform([pred])[0]
            confidence = float(probs[pred])
            return intent, confidence
        except Exception as e:
            logging.error(f"Error classifying intent: {e}")
            return None, 0.0

    def select_response(self, text):
        try:
            X = self.response_vectorizer.transform([text])
            context_vectors = self.response_vectorizer.transform(self.response_contexts)
            similarities = cosine_similarity(X, context_vectors)[0]
            max_sim_idx = np.argmax(similarities)
            response = self.response_texts[max_sim_idx]
            confidence = float(similarities[max_sim_idx])
            return response, confidence
        except Exception as e:
            logging.error(f"Error selecting response: {e}")
            return "Sorry, I couldn't find a suitable response.", 0.0

    def generate_response(self, text):
        try:
            intent, intent_conf = self.classify_intent(text)
            response, response_conf = self.select_response(text)

            # Prioritize intent-based response if confidence is high
            if intent and intent_conf > 0.7:
                base_response = self.response_templates.get(intent, response)
            else:
                base_response = response

            logging.info(f"Input: {text} | Intent: {intent} ({intent_conf:.4f}) | Response: {base_response} ({response_conf:.4f})")
            return {
                "response": base_response,
                "intent": intent,
                "intent_confidence": intent_conf,
                "response_confidence": response_conf
            }
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return {
                "response": "Sorry, I encountered an error. Please try again.",
                "intent": None,
                "intent_confidence": 0.0,
                "response_confidence": 0.0
            }

