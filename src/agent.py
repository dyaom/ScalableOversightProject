"""
Functions to handle calling LLM models
in case we want to make changes later
"""

from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI()
# technically you can `import client` and call this directly?

def completion(
        model: str, messages: str | List[Dict[str,str]],
        **kwargs
    ) -> str:
    if isinstance(messages, str):
        messages = [{'role':'user','content':messages}]
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    ).choices[0].message.content

def parse(
        model: str, messages: List[Dict[str,str]], response_format: BaseModel,
        **kwargs
    ):
    if isinstance(messages, str):
        messages = [{'role':'user','content':messages}]
    return client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        **kwargs
    ).choices[0].message.parsed