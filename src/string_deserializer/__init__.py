__all__ = [
    "StringDeserializer",
    "RegexJsonDeserializer",
    "ImageBytesDeserializer",
    "ImageBytesSaver",
    "MassingCADDeserializer",
    "RosreestrIdParser",
    "DummyRosreestrIdParser",
    "PDFPathImagesCreator",
    "MassingTemplateDeserializer",
    "JsonDeserializer",
    "RegexArgsDeserializer"
]

_LAZY_IMPORTS = {
    "StringDeserializer": (".string_deserializer", "StringDeserializer"),
    "RegexJsonDeserializer": (".regex_json_deserializer", "RegexJsonDeserializer"),
    "ImageBytesDeserializer": (".image_bytes_deserializer", "ImageBytesDeserializer"),
    "ImageBytesSaver": (".image_bytes_saver", "ImageBytesSaver"),
    "MassingCADDeserializer": (".massing_cad_deserializer", "MassingCADDeserializer"),
    "RosreestrIdParser": (".rosreestr_id_parser", "RosreestrIdParser"),
    "DummyRosreestrIdParser": (".dummy_rosreestr_id_parser", "DummyRosreestrIdParser"),
    "PDFPathImagesCreator": (".pdf_path_images_creator", "PDFPathImagesCreator"),
    "MassingTemplateDeserializer": (".massing_template_deserializer", "MassingTemplateDeserializer"),
    "JsonDeserializer": (".json_deserializer", "JsonDeserializer"),
    "RegexArgsDeserializer": (".regex_args_deserializer", "RegexArgsDeserializer")
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")