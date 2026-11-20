"""The only file that talks to a model provider.

To switch providers or models (Groq -> DeepSeek -> a paid model -> a server on SOL),
change the three values below. Nothing else in the project needs to change.
"""
import os
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

load_dotenv()

# --- Change these to switch providers or models ---------------------------
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
MODEL = "openai/gpt-oss-120b"
# ---------------------------------------------------------------------------

# gpt-oss "thinks" before answering, and thinking tokens count against your daily limit.
# "low" keeps it short. Set to None for models that do not support this option.
REASONING_EFFORT = "low"
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
    use_effort = REASONING_EFFORT is not None

    for attempt in range(MAX_ATTEMPTS):
        if use_effort:
            kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
        else:
            kwargs.pop("extra_body", None)
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
            if use_effort:
                print("  [LLM] This model rejected reasoning_effort; retrying without it.")
                use_effort = False
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
