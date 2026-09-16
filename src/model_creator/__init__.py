__all__ = [
    "ModelCreator",
    "HFModelCreator",
    "LoRAModelCreator",
    "PeftAdaptersLoader",
]

_LAZY_IMPORTS = {
    "ModelCreator": (".model_creator", "ModelCreator"),
    "HFModelCreator": (".hf_model_creator", "HFModelCreator"),
    "LoRAModelCreator": (".lora_model_creator", "LoRAModelCreator"),
    "PeftAdaptersLoader": (".peft_adapters_loader", "PeftAdaptersLoader"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
