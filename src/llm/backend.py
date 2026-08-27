"""Unified LLM backend -- the only interface any downstream component may call.

Ported from the notebook's Section 0 `llm_call()` wrapper
(`notebooks/neurosurgery_graphrag_assistant.ipynb`, cell 4). Detects OpenAI
quota/billing failures, disables OpenAI for the rest of the process on the
first such failure (no repeated doomed calls), and transparently falls back
to a local Qwen model -- never silently pretends a failed call succeeded, and
never crashes the caller over a billing issue.

Wrapped behind a `.generate(...)` method so this matches the companion
clinical-rag-eval-harness repo's `LLMBackend` interface convention, in case
the two projects are later factored into a shared internal package.
"""

from __future__ import annotations

from src.config import Config

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, literal research-literature assistant. "
    "Never state a claim as fact unless it is directly supported by the provided context."
)


def is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "insufficient_quota" in msg
        or "rate limit" in msg
        or "429" in msg
        or "exceeded your current quota" in msg
        or "billing" in msg
    )


class OpenAIBackend:
    """LLM backend with an automatic OpenAI-quota -> local-Qwen fallback.

    Once a quota/billing error is observed, OpenAI is disabled for the
    lifetime of this instance and every subsequent call goes straight to the
    local fallback model -- this avoids burning further doomed API calls.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self._client = None
        self._disabled = False
        self._disabled_reason: str | None = None
        self._local_tokenizer = None
        self._local_model = None

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.config.OPENAI_API_KEY or None)
        return self._client

    def _load_local_model(self):
        if self._local_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._local_tokenizer = AutoTokenizer.from_pretrained(self.config.LOCAL_MODEL_ID)
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.config.LOCAL_MODEL_ID, torch_dtype="auto", device_map="auto"
            )
        return self._local_tokenizer, self._local_model

    def _call_openai(
        self, prompt: str, system: str, temperature: float, max_tokens: int, json_mode: bool
    ) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.config.OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
        )
        return resp.choices[0].message.content

    def _call_local(
        self, prompt: str, system: str, temperature: float, max_tokens: int, json_mode: bool
    ) -> str:
        tok, model = self._load_local_model()
        msgs = [
            {
                "role": "system",
                "content": system + (" Respond with valid JSON only, no prose." if json_mode else ""),
            },
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
        return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        system: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 500,
        json_mode: bool = False,
    ) -> str:
        """The single public LLM interface -- every downstream component calls this.

        Tries OpenAI (if `Config.USE_OPENAI` and not already disabled this session). On a
        quota/billing failure, disables OpenAI for the rest of this instance's lifetime and
        falls back to a local Qwen model. A non-quota error is re-raised as a real bug, not
        masked as a fallback trigger.
        """
        if self.config.USE_OPENAI and not self._disabled:
            try:
                return self._call_openai(prompt, system, temperature, max_tokens, json_mode)
            except Exception as e:
                if is_quota_error(e):
                    self._disabled = True
                    self._disabled_reason = str(e)
                else:
                    raise

        return self._call_local(prompt, system, temperature, max_tokens, json_mode)
