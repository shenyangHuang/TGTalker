import os
import json
import tiktoken
from functools import lru_cache
from typing import List, Dict, Union, Iterator, Callable, Optional
from tqdm import tqdm
from loguru import logger

# Cache encodings
_ENCODINGS = {}

@lru_cache(maxsize=128)
def get_encoding(model: str) -> tiktoken.Encoding:
    """Cache and return tiktoken encoding for model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def get_token_count(text: str, model: str) -> int:
    """Count tokens for text string using cached encoding."""
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def get_token_count_from_messages(messages: list, model: str) -> int:
    """
    Count tokens for a list of message dictionaries, including the formatting tokens.
    Implementation based on OpenAI's token counting approach:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4  # Add tokens for the message format
        for key, value in message.items():
            num_tokens +=  get_token_count(str(value), model)
            if key == "name":  # If there's a name, the role is omitted
                num_tokens += -1  # Role is always required and shorter than name

    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens



MODEL_RATES = {
    "gpt-4o-2024-11-20": {"input": 2.5/1000000, "output": 10/1000000},
    "gpt-4o-mini-2024-07-18": {"input": 0.15/1000000, "output": 0.6/1000000},
    "gpt-4.1-2025-04-14": {"input": 2/1000000, "output": 8/1000000},
    "gpt-4.1-mini-2025-04-14": {"input": 0.4/1000000, "output": 1.6/1000000},
    "o4-mini-2025-04-16": {"input": 1.1/1000000, "output": 4.4/1000000},
    "o3-2025-04-16": {"input": 10/1000000, "output": 40/1000000},
    "Qwen/Qwen2.5-7B" : {"input": 0.0, "output": 0.0},
}

def estimate_cost(input_tokens: int, 
                  output_tokens: int, 
                  model: str) -> float:
    """
    Estimate cost based on token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name to determine pricing

    Returns:
        float: Estimated cost in USD
    """
    # GPT-4 pricing (adjust these rates as needed)

    rates = MODEL_RATES.get(model, MODEL_RATES[model])
    return (input_tokens * rates["input"]) + (output_tokens * rates["output"])



def generate_messages(system_prompt: str,
                      messages: List[str]
                      ):
    ''''
    Generates the messages.
    '''
    output = [{"role": "system", "content": system_prompt}]
    for i, message in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        output.append({"role": role, "content": message})
    return output


class ChunkedFileWriter:
    """Handle chunked file writing with automatic cleanup."""

    def __init__(self, folder: str, base_filename: str, chunk_size: int = 10000):
        self.folder = folder
        self.base_filename = base_filename
        self.chunk_size = chunk_size
        self.current_chunk = 1
        self.current_file = None
        os.makedirs(folder, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_file:
            self.current_file.close()

    def write_chunk(self, index: int, data: str):
        if index % self.chunk_size == 0:
            if self.current_file:
                self.current_file.close()
            filename = os.path.join(
                self.folder,
                f"{os.path.splitext(self.base_filename)[0]}_{self.current_chunk}{os.path.splitext(self.base_filename)[1]}"
            )
            self.current_file = open(filename, 'w')
            self.current_chunk += 1
        self.current_file.write(data + "\n")

def write_chunked_files(
    data_iterator: Iterator,
    folder: str,
    base_filename: str,
    model: str,
    output_token_estimate: int = 200,
    output_token_conservative_estimate: int = 400,
    chunk_size: int = 10000,
    create_input_func: Optional[Callable] = None
    ) -> Dict:
    """
    Write data in chunks with token tracking.

    Args:
        data_iterator: Iterator of messages
        folder: folder to save
        base_filename: Base name for output files
        chunk_size: Number of lines per file
        create_input_func: Function to process each item before writing
    """
    total_input_tokens = 0

    with ChunkedFileWriter(folder, base_filename, chunk_size) as writer:
        for i, message in tqdm(enumerate(data_iterator), desc="Processing messages"):
            if isinstance(message, list):
                total_input_tokens += get_token_count_from_messages(message, model)
                line = create_input_func(i, model, message) if create_input_func else message
                writer.write_chunk(i, json.dumps(line))

    summary = {
        "total_input_tokens": total_input_tokens,
        "total_estimated_input_token_cost": estimate_cost(total_input_tokens, 0, model),
        "total_estimated_output_token_cost": estimate_cost(0, i * output_token_estimate, model),
        "total_estimated_conservative_output_token_cost": estimate_cost(0, i * output_token_conservative_estimate, model),
        "total_estimated_cost": estimate_cost(total_input_tokens, i * output_token_estimate, model),
        "total_estimated_conservative_cost": estimate_cost(total_input_tokens, i * output_token_conservative_estimate, model),
        "model": model,
        "total_messages": i
    }

    with open(os.path.join(folder, "token_cost_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(summary)



