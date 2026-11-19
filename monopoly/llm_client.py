"""The only file that talks to the model.

It uses ASU CreateAI through its OpenAI-compatible API (the "Optional: Use the
OpenAI-Compatible SDK" part of the CreateAI tutorial). Put these in the .env file at the
project root (see .env.example):

    CREATEAI_API_KEY=your_project_service_token_here     (the token only, no "Bearer")
    CREATEAI_BASE_URL=https://api-main.aiml.asu.edu      (optional; this is the default)
    LLM_MODEL=openai/gpt4o                               (optional; IDs are "provider/model")

Nothing else in the project needs to change to switch models.
"""
import os
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

load_dotenv()

# --- Settings -------------------------------------------------------------
BASE_URL = os.getenv("CREATEAI_BASE_URL", "https://api-main.aiml.asu.edu").rstrip("/") + "/v1"
API_KEY_ENV = "CREATEAI_API_KEY"
MODEL = os.getenv("LLM_MODEL") or "openai/gpt6_luna"  # lightweight model for testing the agents
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    """The model could not be reached or kept failing."""


class DailyLimitError(LLMError):
    """The free daily quota is used up. Waiting a few seconds will not help."""


_client = None
total_tokens_used = 0
total_calls = 0


def _get_client():
    global _client
    if _client is None:
        key = os.getenv(API_KEY_ENV)
        if not key:
            raise LLMError(f"{API_KEY_ENV} is not set. Put it in your .env file.")
        _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client


def chat(messages, max_tokens=800, temperature=None):
    """Send a list of {"role": ..., "content": ...} messages. Returns the reply text.

    Returns "" if the model sends back nothing (this can happen if its thinking used up
    max_tokens). Counts tokens in total_tokens_used so you can watch your daily budget.
    """
    global total_tokens_used, total_calls
    kwargs = dict(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE if temperature is None else temperature,
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = _get_client().chat.completions.create(**kwargs)
        except RateLimitError as error:
            text = str(error).lower()
            if "per day" in text or "tpd" in text or "rpd" in text:
                raise DailyLimitError(f"Daily limit reached: {error}") from error
            wait = 5 * (2 ** attempt)
            print(f"  [LLM] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        except BadRequestError as error:
            # Different models accept different options. Drop or rename the one the model
            # complained about and try again; anything else is a real error.
            text = str(error).lower()
            if "max_tokens" in text and "max_tokens" in kwargs:
                print("  [LLM] This model wants max_completion_tokens instead of max_tokens; retrying.")
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                continue
            if "temperature" in text and "temperature" in kwargs:
                print("  [LLM] This model rejected the temperature setting; retrying with its default.")
                kwargs.pop("temperature")
                continue
            raise LLMError(f"The model rejected the request: {error}") from error

        total_calls += 1
        if response.usage:
            total_tokens_used += response.usage.total_tokens
        choice = response.choices[0]
        if choice.finish_reason == "length":
            print("  [LLM] Reply was cut off by max_tokens (thinking may have used them up).")
        return choice.message.content or ""

    raise LLMError(f"The model request failed after {MAX_ATTEMPTS} attempts.")


def ask(system_prompt, user_prompt, max_tokens=500):
    """Convenience wrapper for a single question."""
    return chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    print(ask("You are a helpful assistant.", "Say hello in five words."))
    print("Tokens used so far:", total_tokens_used)
