"""
Groq multi-LLM verdict function for Veritas.

Sends a submitted piece of content to 2-3 independent LLMs (hosted on Groq)
and returns each model's score + justification as structured data. This is
the off-chain "judging" step whose results get submitted on-chain via
submit_ai_verdict transactions.
"""

import os
import json
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models used for independent verdicts -- genuinely different architectures,
# not the same model called twice, so disagreement reflects real differences
# in reasoning rather than being theatrical.
EVALUATOR_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

SCORING_PROMPT_TEMPLATE = """You are an impartial evaluator scoring the quality of a submitted piece of content on a blockchain reputation system.

Score the following submission from 0 to 100, where:
- 0-30: Low quality, spam, or unhelpful
- 31-60: Adequate but unremarkable
- 61-85: Good quality, clear value
- 86-100: Excellent, exceptional value

Submission:
\"\"\"
{content}
\"\"\"

Respond with ONLY a JSON object in this exact format, nothing else:
{{"score": <integer 0-100>, "justification": "<one sentence, under 20 words>"}}
"""


@dataclass
class Verdict:
    model: str
    score: int
    justification: str
    error: str = None


def _call_groq_model(model: str, content: str, timeout: int = 20) -> Verdict:
    if not GROQ_API_KEY:
        return Verdict(model=model, score=0, justification="", error="GROQ_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": SCORING_PROMPT_TEMPLATE.format(content=content)}
        ],
        "temperature": 0.3,
        "max_completion_tokens": 500,
    }

    # Qwen3's "thinking" models emit a <think>...</think> reasoning block inline
    # in the content field, which breaks our JSON parsing. reasoning_format="hidden"
    # strips that trace so content only contains the final answer.
    if "qwen" in model.lower():
        payload["reasoning_format"] = "hidden"

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if the model wrapped its JSON in them
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").replace("json\n", "", 1).strip()

        parsed = json.loads(raw_text)
        score = int(parsed["score"])
        score = max(0, min(100, score))  # clamp defensively
        justification = str(parsed["justification"])[:200]

        return Verdict(model=model, score=score, justification=justification)

    except json.JSONDecodeError as e:
        return Verdict(model=model, score=0, justification="", error=f"JSON parse failed on: {raw_text!r}")
    except Exception as e:
        return Verdict(model=model, score=0, justification="", error=str(e))


def get_multi_model_verdicts(content: str, models: list = None) -> list:
    """
    Sends content to multiple Groq-hosted LLMs and returns their verdicts.

    Args:
        content: the submitted text to be scored
        models: optional override of which models to use (defaults to EVALUATOR_MODELS)

    Returns:
        List of Verdict objects, one per model (errors included, not silently dropped)
    """
    models = models or EVALUATOR_MODELS
    return [_call_groq_model(model, content) for model in models]


def check_consensus(verdicts: list, tolerance: int = 15) -> dict:
    """
    Checks whether verdict scores agree within tolerance -- mirrors the
    on-chain is_consensus() logic in contract/contract.py, so this can be
    used for a local pre-check before submitting transactions.
    """
    valid_scores = [v.score for v in verdicts if not v.error]
    if not valid_scores:
        return {"consensus": False, "reason": "no valid verdicts", "scores": []}

    spread = max(valid_scores) - min(valid_scores)
    agreed = spread <= tolerance

    return {
        "consensus": agreed,
        "scores": valid_scores,
        "spread": spread,
        "average": round(sum(valid_scores) / len(valid_scores)),
    }


if __name__ == "__main__":
    sample_content = "This is a well-researched answer that directly addresses the question with clear examples and proper sourcing."

    print("Requesting verdicts from Groq models...\n")
    verdicts = get_multi_model_verdicts(sample_content)

    for v in verdicts:
        if v.error:
            print(f"  {v.model}: ERROR - {v.error}")
        else:
            print(f"  {v.model}: score={v.score}, justification=\"{v.justification}\"")

    result = check_consensus(verdicts)
    print(f"\nConsensus check: {result}")
