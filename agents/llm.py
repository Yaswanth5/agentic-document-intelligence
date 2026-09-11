import json
import os
import time

import ollama

from config import OLLAMA_MODEL


class LocalLLM:
    """
    Local Ollama LLM wrapper.

    Design goals:
    - Keep the model loaded between calls.
    - Keep generated output small.
    - Measure actual inference time.
    - Never retry expensive calls automatically.
    - Return safe empty structures on failure.
    """

    def __init__(self):

        self.model = OLLAMA_MODEL

        self.host = os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434"
        )

        self.client = ollama.Client(
            host=self.host
        )

        self.temperature = float(
            os.getenv(
                "OLLAMA_TEMPERATURE",
                "0"
            )
        )

        # General generation.
        self.num_predict = int(
            os.getenv(
                "OLLAMA_NUM_PREDICT",
                "256"
            )
        )

        # Extraction JSON should be compact.
        self.json_num_predict = int(
            os.getenv(
                "OLLAMA_JSON_NUM_PREDICT",
                "256"
            )
        )

        # Keep model loaded.
        #
        # "-1" means keep loaded indefinitely in Ollama.
        # This is useful for repeated local processing.
        self.keep_alive = os.getenv(
            "OLLAMA_KEEP_ALIVE",
            "-1"
        )

        print(
            f"[LLM] Model: {self.model}"
        )

        print(
            f"[LLM] Host: {self.host}"
        )

        print(
            f"[LLM] keep_alive: {self.keep_alive}"
        )

    # ============================================================
    # GENERATE TEXT
    # ============================================================

    def generate(
        self,
        prompt: str,
        num_predict=None
    ) -> str:

        if not prompt:
            return ""

        start = time.time()

        limit = (
            num_predict
            if num_predict is not None
            else self.num_predict
        )

        print(
            f"[LLM] Starting generation "
            f"prompt_chars={len(prompt)} "
            f"max_output={limit}"
        )

        try:

            response = self.client.chat(
                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                options={
                    "temperature": self.temperature,
                    "num_predict": limit,
                },

                keep_alive=self.keep_alive,
            )

            content = (
                response
                .get("message", {})
                .get("content", "")
                .strip()
            )

            elapsed = time.time() - start

            print(
                f"[LLM] Completed generation "
                f"time={elapsed:.2f}s "
                f"output_chars={len(content)}"
            )

            return content

        except Exception as exc:

            elapsed = time.time() - start

            print(
                f"[LLM] Generation failed "
                f"after {elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}"
            )

            return ""

    # ============================================================
    # GENERATE JSON
    # ============================================================

    def generate_json(
        self,
        prompt: str,
        num_predict=None
    ) -> dict:

        if not prompt:
            return {}

        start = time.time()

        limit = (
            num_predict
            if num_predict is not None
            else self.json_num_predict
        )

        print(
            f"[LLM JSON] Starting "
            f"prompt_chars={len(prompt)} "
            f"max_output={limit}"
        )

        try:

            response = self.client.chat(
                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                # Ollama structured JSON mode.
                format="json",

                options={
                    "temperature": 0,
                    "num_predict": limit,
                },

                keep_alive=self.keep_alive,
            )

            content = (
                response
                .get("message", {})
                .get("content", "")
                .strip()
            )

            elapsed = time.time() - start

            print(
                f"[LLM JSON] Completed "
                f"time={elapsed:.2f}s "
                f"output_chars={len(content)}"
            )

            if not content:
                return {}

            # ----------------------------------------------------
            # Direct JSON
            # ----------------------------------------------------

            try:

                result = json.loads(
                    content
                )

                if isinstance(
                    result,
                    dict
                ):
                    return result

            except json.JSONDecodeError:
                pass

            # ----------------------------------------------------
            # Recover JSON if model added surrounding text.
            # ----------------------------------------------------

            start_pos = content.find(
                "{"
            )

            end_pos = content.rfind(
                "}"
            )

            if (
                start_pos != -1
                and end_pos > start_pos
            ):

                candidate = content[
                    start_pos:
                    end_pos + 1
                ]

                try:

                    result = json.loads(
                        candidate
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

                except json.JSONDecodeError:
                    pass

            print(
                "[LLM JSON] Could not parse JSON"
            )

            return {}

        except Exception as exc:

            elapsed = time.time() - start

            print(
                f"[LLM JSON] Generation failed "
                f"after {elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}"
            )

            return {}