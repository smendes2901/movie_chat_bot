from docarray import BaseDoc
from jina import Executor, requests, Deployment
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel


class PromptDocument(BaseDoc):
    prompt: str
    max_tokens: int


class ModelOutputDocument(BaseDoc):
    token_id: int
    generated_text: str


model_name = "openlm-research/open_llama_3b_v2"


class TokenStreamingExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token="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",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            load_in_8bit=False,
            quantization_config=quantization_config,
            token="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",
        )

        self.model = PeftModel.from_pretrained(
            model, "./openlm-research-open_llama_3b_v2/", torch_dtype=torch.float16
        )

    def starts_with_space(self, token_id):
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return token.startswith("▁")

    @requests(on="/stream")
    async def task(self, doc: PromptDocument, **kwargs) -> ModelOutputDocument:
        input = self.tokenizer(doc.prompt, return_tensors="pt").to("cuda")
        input_len = input["input_ids"].shape[1]

        for output_length in range(doc.max_tokens):
            output = self.model.generate(**input, max_new_tokens=1)
            current_token_id = output[0][-1]
            if current_token_id == self.tokenizer.eos_token_id:
                break

            current_token = self.tokenizer.decode(
                current_token_id, skip_special_tokens=True
            )
            if self.starts_with_space(current_token_id.item()) and output_length > 1:
                current_token = " " + current_token
            yield ModelOutputDocument(
                token_id=current_token_id,
                generated_text=current_token,
            )

            input = {
                "input_ids": output,
                "attention_mask": torch.ones(1, len(output[0])),
            }


if __name__ == "__main__":
    with Deployment(uses=TokenStreamingExecutor, port=1234, protocol="grpc") as dep:
        dep.block()
