#Model registry for sign language word classification
from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.models.model_c import ModelC
MODEL_REGISTRY = {
    "model_a": ModelA,
    "model_b": ModelB,
    "model_c": ModelC,
}
def get_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
