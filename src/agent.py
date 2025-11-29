"""
Functions to handle calling LLM models
in case we want to make changes later
"""

import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Use the common OpenAI client for all endpoints
clients: Dict[str, OpenAI] = {}
if "OPENAI_API_KEY" in os.environ:
    clients["openai"] = OpenAI()
if "GEMINI_API_KEY" in os.environ:
    clients["gemini"] = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"]
    )
if "GROQ_API_KEY" in os.environ:
    clients["groq"] = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"]
    )
if "OPENROUTER_API_KEY" in os.environ:
    clients["openrouter"] = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
# if "HF_TOKEN" in os.environ:
#     clients["hf"] = OpenAI(
#         base_url="https://router.huggingface.co/v1",
#         api_key=os.environ["HF_TOKEN"],
#     )

def _route(model_name: str) -> tuple[str, str]:
    """
    Determine which API to use for a given model name
    
    Parameters
    ----------
    model_name: str
      The model name string
    
    Returns
    -------
    provider: str
      The API provider to use
    name: str
      The cleaned model name string
    """
    try:
        provider, name = model_name.split('/')
    except ValueError:
        raise ValueError(f"Model name {model_name} could not be processed " +
                         "to expected <provider>/<name> format.")
    
    if provider == "openai":
        return "openai", name
    if provider == "google":
        if name.startswith("gemini"):
            return "gemini", name
        # if name.startswith("gemma"):  # only supports some gemma 3 models
        #     return "openrouter", model_name + ":free"
    if provider == "meta-llama":
        if name in ('llama-3.1-8b-instant'): #'llama-3.3-70b-versatile'
            return "groq", name
        if name in ('llama-3.2-3b-instruct', 'llama-3.3-70b-instruct'):
            return "openrouter", model_name + ":free"
    # if provider == "qwen":  # only supports some qwen 3 models
    #     return "openrouter", model_name + ":free"
    
    raise ValueError(f"Model name {model_name} unknown.")

def completion(
        model: str, messages: str | List[Dict[str,str]],
        **kwargs
    ) -> str:

    # get the right API provider
    provider, name = _route(model)
    # check whether we've initialized the client for this provider
    if provider not in clients:
        raise ValueError(
            f"Provider {provider} was not initialized for model {model}. "
            "Please ensure the corresponding API key is set in environment."
        )

    # convert standalone message to proper format, if needed
    if isinstance(messages, str):
        messages = [{'role':'user','content':messages}]
    
    return clients[provider].chat.completions.create(
        model=name,
        messages=messages, # type: ignore
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
    payload = json_str[start_ind:end_ind+1].replace('\n', ' ')
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Input did not contain a single valid JSON string: {e.msg}"
        ) from e
