"""
Model registry for the pattern classifier.

Usage:
    from models import build_model

    model = build_model("cnn1d", in_dim=768, n_labels=9)
    model = build_model("mlp",   in_dim=768, n_labels=9)

Available models:
    mlp    – MLP with mean+max temporal pooling over MERT frames
    cnn1d  – 1D residual CNN that models temporal structure before pooling
"""

from models.mlp   import MLPClassifier
from models.cnn1d import CNN1DClassifier

REGISTRY: dict = {
    "mlp":   MLPClassifier,
    "cnn1d": CNN1DClassifier,
}


def build_model(name: str, in_dim: int, n_labels: int, **kwargs):
    """Instantiate a model by name.

    Args:
        name:     Key in REGISTRY (e.g. "cnn1d", "mlp").
        in_dim:   Feature dimension of each MERT frame (768 for MERT-v1-95M).
        n_labels: Number of output labels.
        **kwargs: Extra keyword args forwarded to the model constructor.
                  Unknown kwargs are silently ignored, so callers can pass a
                  superset without worrying about per-model parameter names.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](in_dim=in_dim, n_labels=n_labels, **kwargs)
