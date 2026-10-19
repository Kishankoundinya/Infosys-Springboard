import os
import threading
from typing import Optional, List, Dict, Any
from backend.app.utils.config import settings
from backend.app.utils.logging import logger
from backend.app.rag.embeddings import get_device

class SharedLLM:
    """
    Singleton local LLM manager.
    Loads TinyLlama once into memory and provides thread-safe inference for all 6 agents.
    """
    _instance: Optional["SharedLLM"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_name = settings.LLM_MODEL
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._is_loaded = False
        self._load_lock = threading.Lock()
        self._gen_cache: Dict[str, str] = {}

    @classmethod
    def get_instance(cls) -> "SharedLLM":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self):
        with self._load_lock:
            if self._is_loaded:
                return

            logger.info(f"Loading Shared LLM: {self.model_name} on device: {self.device}...")
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

                # Optimize CPU thread allocation to prevent laptop thread thrashing
                if self.device == "cpu":
                    try:
                        torch.set_num_threads(min(4, os.cpu_count() or 4))
                    except Exception:
                        pass

                torch_dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                    )

                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )

                if self.device != "auto" and hasattr(self.model, "to"):
                    try:
                        self.model.to(self.device)
                    except Exception as e:
                        logger.warning(f"Could not move model to {self.device}, using CPU: {e}")
                        self.device = "cpu"
                        self.model.to("cpu")

                self.model.eval()
                self._is_loaded = True
                logger.info(f"Shared LLM '{self.model_name}' loaded successfully on {self.device} in eval mode.")
            except Exception as e:
                logger.error(f"Failed to load Shared LLM '{self.model_name}': {e}")
                self._is_loaded = False
                raise RuntimeError(f"Error loading LLM {self.model_name}: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate grounded text using TinyLlama's chat format with accelerated inference.
        """
        if not self._is_loaded:
            self.load_model()

        max_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
        temp = temperature if temperature is not None else settings.TEMPERATURE

        sys_msg = system_prompt or "You are a financial document analysis assistant. Use only the supplied source context. Do not use unsupported knowledge."

        formatted_prompt = (
            f"<|system|>\n{sys_msg}</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            f"<|assistant|>\n"
        )

        cache_key = f"{formatted_prompt}_{max_tokens}_{temp}"
        if cache_key in self._gen_cache:
            logger.info("Serving LLM output from in-memory inference cache (0ms).")
            return self._gen_cache[cache_key]

        try:
            import torch
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            if self.device in ["cuda", "mps"]:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # High performance inference: inference_mode + KV caching + greedy decoding
            use_sampling = temp > 0.3
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": use_sampling,
            }
            if use_sampling:
                gen_kwargs["temperature"] = temp

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Slice out the input tokens to get just the generated assistant reply
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if len(self._gen_cache) > 256:
                self._gen_cache.clear()
            self._gen_cache[cache_key] = response
            return response
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            raise e

    def get_langchain_llm(self) -> "LocalTinyLlamaLLM":
        return LocalTinyLlamaLLM()

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class LocalTinyLlamaLLM(LLM):
    """LangChain-native LLM wrapper for Shared TinyLlama instance."""
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        shared = SharedLLM.get_instance()
        return shared.generate(prompt=prompt, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "tinyllama_local"

def get_shared_llm() -> SharedLLM:
    return SharedLLM.get_instance()

def get_langchain_llm() -> LocalTinyLlamaLLM:
    return LocalTinyLlamaLLM()
