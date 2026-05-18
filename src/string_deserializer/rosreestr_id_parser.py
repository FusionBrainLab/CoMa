from abc import ABC, abstractmethod
from typing import Any
import re
import json

import requests

from .string_deserializer import StringDeserializer
from ..string_formatter import StringFormatter

class RosreestrIdParser(StringDeserializer):
    def __init__(self, *, url_template: str,
                        ca_cert_path: str,
                        client_cert_path: str,
                        radius: int) -> None:
        self.url_template = url_template
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.string_formatter = StringFormatter()

    def __call__(self, *, string: str) -> Any:
        url = self.string_formatter(string=self.url_template, args={"cadastralNumber":string})
        response = requests.get(url, verify=self.ca_cert_path, cert=self.client_cert_path)
        return response.json()