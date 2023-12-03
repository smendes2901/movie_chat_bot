import asyncio
from docarray import BaseDoc
from jina import Client


class PromptDocument(BaseDoc):
    prompt: str
    max_tokens: int


class ModelOutputDocument(BaseDoc):
    token_id: int
    generated_text: str


llama_prompt = PromptDocument(
    prompt="""
<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Always answer as helpfully 
as possible while being safe.  Your answers should not include any harmful, 
unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure 
that your responses are socially unbiased and positive in nature.

If a question does not make any sense or is not factually coherent, explain why 
instead of answering something incorrectly. If you don't know the answer to a 
question, don't share false information.
<</SYS>>
If I punch myself in the face and it hurts, am I weak or strong? [/INST]
""",
    max_tokens=100,
)


async def main():
    client = Client(
        port=8000,
        host="https://fvdvu51ug7j-496ff2e9c6d22116-8000-colab.googleusercontent.com/",
        asyncio=True,
    )
    async for doc in client.stream_doc(
        on="/stream",
        inputs=llama_prompt,
        return_type=ModelOutputDocument,
    ):
        print(doc.generated_text, end="")


asyncio.run(main())
