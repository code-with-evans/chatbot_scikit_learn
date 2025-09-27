import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from api.extract_data import load_clinc150, load_daily_dialog, preprocess_and_vectorize, encode_labels, save_preprocessors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_confidence_scores(model, X):
    """Compute average confidence score for the top predicted class."""
    try:
        probs = model.predict_proba(X)
        top_probs = np.max(probs, axis=1)
        return np.mean(top_probs)
    except Exception as e:
        logging.error(f"Error computing confidence scores: {e}")
        return 0.0

def compute_response_similarity(contexts, responses, vectorizer):
    """Compute average cosine similarity for response selection on a validation set."""
    try:
        split = int(0.8 * len(contexts))
        X_train = vectorizer.transform(contexts[:split])
        X_test = vectorizer.transform(contexts[split:])
        similarities = []
        for i, x_test in enumerate(X_test):
            sims = cosine_similarity(x_test, X_train)[0]
            max_sim = np.max(sims)
            similarities.append(max_sim)
        return np.mean(similarities)
    except Exception as e:
        logging.error(f"Error computing response similarity: {e}")
        return 0.0

def train_intent_model():
    try:
        train_texts, train_labels, test_texts, test_labels = load_clinc150()
        X_train, vectorizer = preprocess_and_vectorize(train_texts, fit_vectorizer=True)
        y_train, encoder = encode_labels(train_labels, fit_encoder=True)
        save_preprocessors(vectorizer, encoder, 'intent')

        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Intent model trained.")

        # Evaluate on test set
        X_test = preprocess_and_vectorize(test_texts, vectorizer=vectorizer)
        y_test = encode_labels(test_labels, encoder=encoder)
        acc = accuracy_score(y_test, model.predict(X_test))
        avg_conf = compute_confidence_scores(model, X_test)
        logging.info(f"Intent model - Accuracy: {acc:.4f}, Avg Confidence: {avg_conf:.4f}")

        joblib.dump(model, 'models/intent_model.pkl')
        return model, vectorizer, encoder
    except Exception as e:
        logging.error(f"Error training intent model: {e}")
        raise

def train_response_model(contexts, responses):
    try:
        # Vectorize contexts for response selection
        X, vectorizer = preprocess_and_vectorize(contexts, fit_vectorizer=True)
        save_preprocessors(vectorizer, name='response')

        # Save contexts and responses for retrieval
        joblib.dump(contexts, 'models/response_contexts.pkl')
        joblib.dump(responses, 'models/response_texts.pkl')
        logging.info("Response model data saved.")

        # Evaluate similarity on validation set
        avg_sim = compute_response_similarity(contexts, responses, vectorizer)
        logging.info(f"Response model - Avg Cosine Similarity: {avg_sim:.4f}")

        return vectorizer
    except Exception as e:
        logging.error(f"Error training response model: {e}")
        raise

if __name__ == "__main__":
    train_intent_model()
    contexts, responses = load_daily_dialog()
    train_response_model(contexts, responses)
