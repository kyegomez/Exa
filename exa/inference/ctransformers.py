from types import NoneType
from typing import Generator, Optional, Sequence, Union


from ctransformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers.hub import AutoConfig
from ctransformers.llm import Config


class CInference:
    def __init__(
        self,
        model_path: str = None,
        model_type: Optional[str] = None,
        model_file: Optional[str] = None,
        config: Optional[Union[AutoConfig, Config]] = None,
        lib: Optional[str] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        hf: bool = False,
        **kwargs
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            model_file=model_file,
            config=config,
            lib=lib,
            local_files_only=local_files_only,
            revision=revision,
            hf=hf,
            **kwargs
        )
        if hf:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = None
    def run(
            self, 
            prompt: str,
            max_new_tokens: Optional[int] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            temperature: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            last_n_tokens: Optional[int] = None,
            seed: Optional[int] = None,
            batch_size: Optional[int] = None,
            threads: Optional[int] = None,
            stop: Optional[Sequence[str]] = None,
            stream: Optional[bool] = True,
            reset: Optional[bool] = None
        ) -> Union[str, Generator[str, NoneType, NoneType]]:
        if self.tokenizer:
            return self.model.__call__(
                prompt,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                last_n_tokens=last_n_tokens,
                seed=seed,
                batch_size=batch_size,
                threads=threads,
                stop=stop,
                stream=stream,
                reset=reset
            )
        else:
            tokens = self.model.tokenize(prompt)
            return self.model.generate(
                tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                last_n_tokens=last_n_tokens,
                seed=seed,
                batch_size=batch_size,
                threads=threads,
                reset=reset
            )