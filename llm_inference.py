import torch
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    TextStreamer,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class LLM:
    def __init__(self, model_name, lora_path):
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            token="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",
            device_map="cuda",
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            token="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",
        )

        self.model = PeftModel.from_pretrained(
            model, lora_path, torch_dtype=torch.float16
        )

        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    def generate(self, prompt):
        # print(prompt)
        torch.cuda.empty_cache()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
                streamer=self.streamer,
            )

        s = generation_output.sequences[0]
        return self.tokenizer.decode(s, skip_special_tokens=True)
