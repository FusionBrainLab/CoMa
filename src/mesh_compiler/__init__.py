__all__ = [
    "MeshCompiler",
    "MassingMeshCompiler",
    "MassingContextMeshCompiler"
]

_LAZY_IMPORTS = {
    "MeshCompiler": (".mesh_compiler", "MeshCompiler"),
    "MassingMeshCompiler": (".massing_mesh_compiler", "MassingMeshCompiler"),
    "MassingContextMeshCompiler": (".massing_context_mesh_compiler", "MassingContextMeshCompiler")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")