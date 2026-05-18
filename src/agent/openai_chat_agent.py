from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Protocol, Any
from dataclasses import dataclass
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from copy import deepcopy

from openai import OpenAI
from tqdm import tqdm

from .agent import Agent
from ..agent_message import AgentMessage, AgentMessageContent
from ..memory_to_dict_parser import MemoryToDictParser

def get_response(
    i,
    data, 
    api_key,
    base_url,
    client_kwargs,
    model_name,
    request_kwargs
):
    messages, tools = data
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        **client_kwargs
    )
    response = None
    while response == None:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                **request_kwargs,
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
    return (i, response)

class OpenAIChatAgent(Agent):
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

    def __call__(self, *, memories: List[List[AgentMessage]]) -> List[List[AgentMessage]]:
        dict_messages = []
        functions = []
        for memory in memories:
            local_dict = self.memory_to_dict_parser(memory=memory)
            dict_messages.append(local_dict[self.messages_key])
            functions.append(local_dict.get(self.tools_key, []))

        response_fn = get_response
        data = [(dict_messages[i], functions[i]) for i in range(len(dict_messages))]

        results = {}
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_item = {executor.submit(get_response, i, d, self.api_key,self.base_url,self.client_kwargs,self.model_name,self.request_kwargs): d for i, d in enumerate(data)}
            for future in tqdm(as_completed(future_to_item), total=len(data)):
                result = future.result()
                results[result[0]] = result[1]
        responses = []
        for i in range(len(data)):
            responses.append(results[i])

        """with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            responses = list(executor.map(response_fn, data))"""
        """responses = []
        for i in tqdm(range(len(dict_messages))):
            responses.append(response_fn(data[i]))"""

        answers = []
        for r in responses:
            if type(r) == type("string"):
                c = [AgentMessageContent(modality="text", content=r)]
                answers.append([AgentMessage(role="assistant", content=c)])
                continue
            try:
                content = [AgentMessageContent(modality="text", content=r.choices[0].message.content.strip())]
            except:
                content = []
            if r.choices[0].message.tool_calls != None:
                for tool_call in r.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except:
                        tool_args = {}
                    tool_dict = {"name":tool_name, "arguments":tool_args, "id":tool_call.id}
                    function_call_content = json.dumps(tool_dict, ensure_ascii=False)
                    content.append(AgentMessageContent(modality="function_call", content=function_call_content))
            answers.append([AgentMessage(role="assistant", content=content)])
        return answers