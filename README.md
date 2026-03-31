# How to run?
# Extract once (saves data/embeddings_all.pt)
python scripts/extract_embeddings.py --device cuda

# Train with holdout (default)
python scripts/train_classifier.py

# Train with 5-fold cross-validation
python scripts/train_classifier.py --split-strategy kfold --n-folds 5

# Evaluate on held-out test set
python scripts/evaluate.py