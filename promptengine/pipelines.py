from abc import abstractmethod
from typing import List, Dict, Tuple, Iterator
import json
import os
from promptengine.utils import LLM, call_chatgpt, is_valid_filepath, is_valid_json
from promptengine.template import PromptTemplate, PromptPermutationGenerator


class PromptPipeline:
    def __init__(self, storageFile: str):
        if not is_valid_filepath(storageFile):
            raise IOError(f"Filepath {storageFile} is invalid, or you do not have write access.")

        self._filepath = storageFile

    @abstractmethod
    def gen_prompts(self, properties) -> List[PromptTemplate]:
        raise NotImplementedError("Please Implement the gen_prompts method")
    
    @abstractmethod
    def analyze_response(self, response) -> bool:
        """
            Analyze the response and return True if the response is valid.
        """
        raise NotImplementedError("Please Implement the analyze_response method")
    
    def gen_responses(self, properties, llm: LLM, n: int = 1, temperature: float = 1.0) -> Iterator[Dict]:
        """
            Calls LLM 'llm' with all prompts, and yields responses as dicts in format {prompt, query, response, llm, info}.

            By default, for each response, this also saves reponses to disk as JSON at the filepath given during init. 
            (Very useful for saving money in case something goes awry!)
            To clear the cached responses, call clear_cached_responses(). 

            Do not override this function.
        """
        # Double-check that properties is the correct type (JSON dict):
        if not is_valid_json(properties):
            raise ValueError(f"Properties argument is not valid JSON.")

        # Generate concrete prompts using properties dict
        prompts = self.gen_prompts(properties)

        # Load any cache'd responses
        responses = self._load_cached_responses()

        # Query LLM with each prompt, yield + cache the responses
        for prompt in prompts:
            if isinstance(prompt, PromptTemplate) and not prompt.is_concrete():
                raise Exception(f"Cannot send a prompt '{prompt}' to LLM: Prompt is a template.")
            
            # Each prompt has a history of what was filled in from its base template.
            # This data --like, "class", "language", "library" etc --can be useful when parsing responses.
            info = prompt.fill_history
            prompt_str = str(prompt)
            
            # First check if there is already a response for this item. If so, we can save an LLM call:
            if prompt_str in responses:
                print(f"   - Found cache'd response for prompt {prompt_str}. Using...")
                yield {
                    "prompt": prompt_str,
                    "query": responses[prompt_str]["query"],
                    "response": responses[prompt_str]["response"],
                    "llm": responses[prompt_str]["llm"] if "llm" in responses[prompt_str] else LLM.ChatGPT.name,
                    "info": responses[prompt_str]["info"],
                }
                continue

            # Call the LLM to generate a response
            query, response = self._prompt_llm(llm, prompt_str, n, temperature)

            # Save the response to a JSON file
            # NOTE: We do this to save money --in case something breaks between calls, can ensure we got the data!
            responses[prompt_str] = {
                "query": query, 
                "response": response,
                "llm": llm.name,
                "info": info,
            }
            self._cache_responses(responses)

            yield {
                "prompt":prompt_str, 
                "query":query, 
                "response":response,
                "llm": llm.name,
                "info": info,
            }
    
    def _load_cached_responses(self) -> Dict:
        """
            Loads saved responses of JSON at self._filepath. 
            Useful for continuing if computation was interrupted halfway through. 
        """
        if os.path.isfile(self._filepath):
            with open(self._filepath, encoding="utf-8") as f:
                responses = json.load(f)
            return responses
        else:
            return {}
    
    def _cache_responses(self, responses) -> None:
        with open(self._filepath, "w") as f:
            json.dump(responses, f)
    
    def clear_cached_responses(self) -> None:
        self._cache_responses({})

    def _prompt_llm(self, llm: LLM, prompt: str, n: int = 1, temperature: float = 1.0) -> Tuple[Dict, Dict]:
        if llm is LLM.ChatGPT:
            return call_chatgpt(prompt, n=n, temperature=temperature)
        else:
            raise Exception(f"Language model {llm} is not supported.")

