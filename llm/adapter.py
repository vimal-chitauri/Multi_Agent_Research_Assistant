import os
from dotenv import load_dotenv

load_dotenv(override=True)


class LLMAdapter:

    ANTHROPIC_MODELS = {
        "fast":  "claude-haiku-4-5-20251001",
        "smart": "claude-sonnet-4-6",
        "mix":   "claude-sonnet-4-6",
    }

    GROQ_MODELS = {
        "fast":  "llama-3.1-8b-instant",
        "smart": "llama-3.3-70b-versatile",
        "mix":   "meta-llama/llama-4-scout-17b-16e-instruct",
    }

    GROQ_FALLBACK = {
        "llama-3.3-70b-versatile":                   "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct": "llama-3.1-8b-instant",
        "llama-3.1-8b-instant":                      None,
    }

    def __init__(self, model: str = "fast"):
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._groq_key      = os.getenv("GROQ_API_KEY", "")

        if self._anthropic_key:
            self._provider = "anthropic"
            self.model     = self.ANTHROPIC_MODELS.get(model, model)
            import anthropic as _ant
            self._client   = _ant.Anthropic(api_key=self._anthropic_key)
        elif self._groq_key:
            self._provider = "groq"
            self.model     = self.GROQ_MODELS.get(model, model)
            from groq import Groq
            self._client   = Groq(api_key=self._groq_key)
        else:
            raise RuntimeError(
                "No LLM API key found. Set ANTHROPIC_API_KEY or GROQ_API_KEY in .env"
            )


    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, _model: str = None) -> str:
        model = _model or self.model
        if self._provider == "anthropic":
            return self._anthropic_chat(system_prompt, user_message, temperature, model)
        return self._groq_chat(system_prompt, user_message, temperature, model)

    def chat_with_history(self, messages: list[dict], _model: str = None) -> str:
        model = _model or self.model
        if self._provider == "anthropic":
            return self._anthropic_history(messages, model)
        return self._groq_history(messages, model)


    def _anthropic_chat(self, system_prompt: str, user_message: str,
                        temperature: float, model: str) -> str:
        response = self._client.messages.create(
            model      = model,
            max_tokens = 4096,
            temperature= temperature,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _anthropic_history(self, messages: list[dict], model: str) -> str:
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        response = self._client.messages.create(
            model      = model,
            max_tokens = 4096,
            temperature= 0.7,
            system     = system,
            messages   = filtered,
        )
        return response.content[0].text


    def _groq_chat(self, system_prompt: str, user_message: str,
                   temperature: float, model: str) -> str:
        try:
            r = self._client.chat.completions.create(
                model      = model,
                messages   = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature= temperature,
            )
            return r.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                fallback = self.GROQ_FALLBACK.get(model)
                if fallback:
                    print(f"[LLMAdapter] {model} rate-limited → {fallback}")
                    return self._groq_chat(system_prompt, user_message, temperature, fallback)
            raise

    def _groq_history(self, messages: list[dict], model: str) -> str:
        try:
            r = self._client.chat.completions.create(
                model      = model,
                messages   = messages,
                temperature= 0.7,
            )
            return r.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                fallback = self.GROQ_FALLBACK.get(model)
                if fallback:
                    print(f"[LLMAdapter] {model} rate-limited → {fallback}")
                    return self._groq_history(messages, fallback)
            raise
