import streamlit as st
import json
import threading
from typing import Any, Optional, Literal, List, Dict
from dspy import OpenAI

# Load secrets from .streamlit/secrets.toml
secrets = st.secrets

# OpenAI API key from secrets
OPENAI_API_KEY = secrets["openai"]["api_key"]

class OpenAIModel(OpenAI):
    """
    A wrapper class for dspy.OpenAI that adds token usage logging.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAIModel.

        Args:
            model: The name of the OpenAI model to use.
            api_key: The OpenAI API key.
            model_type: The type of the model ("chat" or "text").
            **kwargs: Additional keyword arguments to pass to dspy.OpenAI.
        """
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """
        Log the total tokens used from the OpenAI API response.

        Args:
            response: The response from the OpenAI API.
        """
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """
        Get the total tokens used and reset the token usage counters.

        Returns:
            A dictionary containing the prompt and completion token usage for the model.
        """
        usage = {
            self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Query the OpenAI model and track token usage.

        Args:
            prompt: The prompt to send to the OpenAI model.
            only_completed: Whether to return only completed choices.
            return_sorted: Whether to return the choices sorted by score.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            A list of completions from the OpenAI model.
        """
        response = self.request(prompt, **kwargs)
        self.log_usage(response)
        choices = response["choices"]
        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = sorted(
                [
                    (
                        sum(c["logprobs"]["token_logprobs"]) / len(c["logprobs"]["token_logprobs"]),
                        self._get_choice_text(c),
                    )
                    for c in choices
                ],
                reverse=True,
            )
            completions = [c for _, c in scored_completions]

        return completions


# Initialize the OpenAI model
openai_model = OpenAIModel(api_key=OPENAI_API_KEY)


def generate_sql_query(salesforce_data: Dict, requirement: str) -> str:
    """Generates a SQL query using OpenAI based on the Salesforce data and requirement.

    Args:
        salesforce_data (Dict): The Salesforce object and field data.
        requirement (str): The user's requirement for the query.

    Returns:
        str: The generated SQL query.
    """

    object_list = salesforce_data["object_list"]
    object_tree = salesforce_data["object_tree"]

    prompt = f"""You are an expert at creating SQL queries for Salesforce.
    
    The following are the objects available: {object_list}
    
    Here is an example of fields available for the 'Account' object: {object_tree['Account']}
    
    Create a SQL query that satisfies the following requirement: {requirement}"""

    response = openai_model(prompt)
    return response[0]


# Load the JSON data from file
with open('salesforce_object_data.json', 'r') as f:
    salesforce_data = json.load(f)

# Example usage: Generate a SQL query to get all accounts with AnnualRevenue greater than 1 million
requirement = "Get all accounts with AnnualRevenue greater than 1 million"
sql_query = generate_sql_query(salesforce_data, requirement)
print(sql_query)