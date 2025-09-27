Scikit-Learn Chatbot
A production-ready chatbot built with scikit-learn and spaCy, designed for semantic intent discovery using embeddings and clustering. The chatbot processes dialogues from the DailyDialog dataset, discovers intents via KMeans clustering, and uses an ensemble of Logistic Regression, LinearSVC, and Random Forest for intent classification, with Nearest Neighbors for response retrieval. It includes a FastAPI endpoint for scalable deployment.

Features
Semantic Intent Discovery: Uses KMeans clustering on spaCy embeddings (en_core_web_lg) for fully semantic intent detection, avoiding rule-based methods.
Machine Learning Models: Ensemble classifier (Logistic Regression, LinearSVC, Random Forest) for intent classification, Nearest Neighbors for response retrieval.
Dimensionality Reduction: Applies TruncatedSVD to reduce embedding dimensions for efficient training.

Production-Ready:
Modular pipeline with logging and error handling.
Batch processing with nlp.pipe for efficiency.
Parallel training with n_jobs=-1.
FastAPI for scalable API deployment.
Similarity threshold (0.8) for response quality.
Intent balancing via undersampling.
