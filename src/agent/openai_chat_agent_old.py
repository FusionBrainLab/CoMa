from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Protocol, Any
from dataclasses import dataclass
import json
import math
from concurrent.futures import ThreadPoolExecutor
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import threading

from openai import OpenAI
from tqdm import tqdm

from .agent import Agent
from ..agent_message import AgentMessage, AgentMessageContent
from ..memory_to_dict_parser import MemoryToDictParser

class OpenAIChatAgentOld(Agent):
    def __init__(self, *, api_key_var: str,
                        base_url: str,
                        model_name: str,
                        request_kwargs: Dict[str, Any],
                        max_workers: int,
                        memory_to_dict_parser: MemoryToDictParser,
                        messages_key: str,
                        tools_key: str,
                        client_kwargs: Dict[str, Any]) -> None:
        self.api_key = os.getenv(api_key_var)
        self.base_url = base_url
        self.model_name = model_name
        self.request_kwargs = request_kwargs
        self.n_workers = min(os.cpu_count(), max_workers)
        self.memory_to_dict_parser = memory_to_dict_parser
        self.messages_key = messages_key
        self.tools_key = tools_key
        self.client_kwargs = client_kwargs
        log_dir = Path("/home/jovyan/SR008.fs2/maslov/massing_generation/experiments/baselines/240226_procedural")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.response_log_path = log_dir / "llm_answers.log"
        self._log_lock = threading.Lock()

    def __call__(self, *, memories: List[List[AgentMessage]]) -> List[List[AgentMessage]]:
        dict_messages = []
        functions = []
        for memory in memories:
            local_dict = self.memory_to_dict_parser(memory=memory)
            dict_messages.append(local_dict[self.messages_key])
            functions.append(local_dict.get(self.tools_key, []))

        def get_response(data):
            messages, tools = data
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self.client_kwargs
            )
            response = None
            while response == None:
                try:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=tools,
                        **self.request_kwargs,
                    )
                except Exception as e:
                    r = repr(e)
                    if "BadRequestError" in r:
                        response = r
                        break
                    else:
                        print("OpenAIChatAgent generation failed. Retrying...")
                        print(repr(e))
                        continue
            return response
        response_fn = get_response
        data = [(dict_messages[i], functions[i]) for i in range(len(dict_messages))]

        """with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            responses = list(executor.map(response_fn, data))"""
        responses = []
        for i in tqdm(range(len(dict_messages))):
            responses.append(response_fn(data[i]))

        answers = []
        for idx, r in enumerate(responses):
            if isinstance(r, str):
                content = [AgentMessageContent(modality="text", content=r)]
            else:
                try:
                    content = [AgentMessageContent(modality="text", content=r.choices[0].message.content.strip())]
                except Exception:
                    content = []
                if r.choices[0].message.tool_calls is not None:
                    for tool_call in r.choices[0].message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except Exception:
                            tool_args = {}
                        tool_dict = {"name": tool_name, "arguments": tool_args, "id": tool_call.id}
                        function_call_content = json.dumps(tool_dict, ensure_ascii=False)
                        content.append(AgentMessageContent(modality="function_call", content=function_call_content))
            answers.append([AgentMessage(role="assistant", content=content)])
            self._log_answer(
                messages=dict_messages[idx],
                response_contents=self._serialize_contents(contents=content),
                request_index=idx,
            )

        return answers


    def _serialize_contents(self, *, contents: List[AgentMessageContent]) -> List[Dict[str, Any]]:
        return [{"modality": item.modality, "content": item.content} for item in contents]

    def _log_answer(self, *, messages: List[Dict[str, Any]], response_contents: List[Dict[str, Any]], request_index: int) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": self.model_name,
            "request_index": request_index,
            "messages": messages,
            "response": response_contents,
        }
        with self._log_lock:
            with self.response_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
