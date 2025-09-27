import logging
import os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_clinc150():
    try:
        dataset = load_dataset("clinc_oos", "plus")
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['intent']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['intent']
        logging.info("CLINC150 dataset loaded successfully.")
        return train_texts, train_labels, test_texts, test_labels
    except Exception as e:
        logging.error(f"Error loading CLINC150: {e}")
        raise

def load_daily_dialog():
    try:
        dataset = load_dataset("frankdarkluo/DailyDialog")
        contexts = []
        responses = []
        for split in ['train', 'validation', 'test']:
            for context, response in zip(dataset[split]['context'], dataset[split]['response']):
                contexts.append(context.strip())
                responses.append(response.strip())
        logging.info(f"DailyDialog dataset loaded successfully. {len(contexts)} context-response pairs loaded.")
        return contexts, responses
    except Exception as e:
        logging.error(f"Error loading DailyDialog: {e}")
        raise

def preprocess_and_vectorize(texts, fit_vectorizer=False, vectorizer=None):
    try:
        if fit_vectorizer:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)
            return X, vectorizer
        else:
            return vectorizer.transform(texts)
    except Exception as e:
        logging.error(f"Error in vectorization: {e}")
        raise

def encode_labels(labels, fit_encoder=False, encoder=None):
    try:
        if fit_encoder:
            encoder = LabelEncoder()
            y = encoder.fit_transform(labels)
            return y, encoder
        else:
            return encoder.transform(labels)
    except Exception as e:
        logging.error(f"Error in label encoding: {e}")
        raise

def save_preprocessors(vectorizer, encoder=None, name=""):
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, f'api/models/{name}_vectorizer.pkl')
        if encoder:
            joblib.dump(encoder, f'api/models/{name}_encoder.pkl')
        logging.info(f"Preprocessors for {name} saved.")
    except Exception as e:
        logging.error(f"Error saving preprocessors: {e}")
        raise