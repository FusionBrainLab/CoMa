from abc import ABC, abstractmethod
from typing import Any
import re
import json
import io

from PIL import Image
import fitz

from .string_deserializer import StringDeserializer

class PDFPathImagesCreator(StringDeserializer):
    def __call__(self, *, string: str) -> Any:
        images = []
    
        pdf_document = fitz.open(string)
        
        zoom = 144 / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None

            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            images.append(img)
        
        pdf_document.close()
        return images