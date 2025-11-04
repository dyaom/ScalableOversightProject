"""
Functions to handle calling LLM models
in case we want to make changes later
"""

import json
from typing import Dict, List, Union

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# openai_client = OpenAI()
# hf_client = InferenceClient()
clients: Dict[str, Union[OpenAI, InferenceClient]] = {
    "openai": OpenAI(),
    "hf": InferenceClient()
}

def _route(model: str):
    """
    Determine which API to use for a given model name
    
    Parameters
    ----------
    model: str
      The model name string
    
    Returns
    -------
    provider: str
      The API provider to use
    name: str
      The cleaned model name string
    """
    if model.startswith("openai/"):
        return "openai", model[7:]
    return "hf", model

def completion(
        model: str, messages: str | List[Dict[str,str]],
        **kwargs
    ) -> str:
    # get the right API provider
    provider, name = _route(model)
    # convert standalone message to proper format, if needed
    if isinstance(messages, str):
        messages = [{'role':'user','content':messages}]
    
    return clients[provider].chat.completions.create(
        model=name,
        messages=messages,
        **kwargs
    ).choices[0].message.content

# def parse(
#         model: str, messages: List[Dict[str,str]], response_format: BaseModel,
#         **kwargs
#     ) -> Dict:
#     provider, name = _route(model)
#     if isinstance(messages, str):
#         messages = [{'role':'user','content':messages}]
    
#     if provider == "openai":
#         response = clients["openai"].beta.chat.completions.parse(
#             model=name,
#             messages=messages,
#             response_format=response_format,
#             **kwargs
#         ).choices[0].message.parsed
#         return dict(response)
#     elif provider == "hf":  # NOT SUPPORTED FOR ALL HF MODELS
#         response_dict = {
#             "type": "json_schema",
#             "json_schema": {
#                 "name": response_format.__name__,
#                 "schema": response_format.model_json_schema(),
#                 "strict": True
#             }
#         }
#         response = clients["hf"].chat_completion(
#             model=name,
#             messages=messages,
#             response_format=response_dict,
#             **kwargs
#         )
#         return json.loads(response)
    
def read_json(json_str: str) -> Dict:
    """
    Convert a LLM-provided string containing a JSON object to a dictionary
    
    Parameters
    ----------
    json_str: str
        The JSON-containing string
    
    Returns
    -------
    payload: dict
        The JSON content as a dictionary

    """
    start_ind = json_str.find('{')
    end_ind = json_str.rfind('}')
    payload = json_str[start_ind:end_ind+1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Input did not contain a single valid JSON string: {e.msg}"
        ) from e
