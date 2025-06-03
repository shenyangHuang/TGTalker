import argparse
import json
import numpy as np
import pandas as pd
import os
import torch


from enum import Enum
from glob import glob
from outlines import models, generate, samplers
from pydantic import BaseModel, constr
from transformers import AutoTokenizer
from tqdm import tqdm


#################
## JSON Schema ##


TGBExplanation = json.dumps(
    {
        "type": "object",
        "properties": {"explanation": {"type": "string"}},
        "required": ["explanation"],
    }
)


class TGBExplanationCat(str, Enum):
    most_recent = "Most Recent Interaction Heuristic"
    repeated_interaction = "Repeated / Consistent Interaction Pattern"
    pattern_continuation = "Pattern Continuation or Extrapolation"
    lack_of_data = "Lack of Data or Default / Fallback Prediction"
    new_node = "New Node or Unseen Interaction"
    sequence_logic = "Sequence or Alternation Logic"
    most_frequent = "Most Frequent Past Destination"
    ambiguous = "Ambiguous or Multiple Candidates"
    default = "Default or Most Common Node"
    others = "Others"


#################
def sanitize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def make_model_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )


def make_explanation_messages(messages, destination_node):

    assistant_message = {
        "role": "assistant",
        "content": "{" + f'"destination_node":{destination_node}' + "}",
    }
    messages.append(assistant_message)
    user_message = {
        "role": "user",
        "content": "Please provide a reasoning for your prediction.",
    }
    messages.append(user_message)
    return messages


def make_explanation_category_messages(system_prompt, explanation):

    system_message = {
        "role": "system",
        "content": system_prompt,
    }
    user_message = {
        "role": "user",
        "content": explanation,
    }
    return [system_message, user_message]


if __name__ == "__main__":

    parser = argparse.ArgumentParser("*** arguments for running LLM ***")

    parser.add_argument(
        "--answer_cache",
        action="store_true",
        help="cache if answer_key is calculated",
        default=False,
    )

    parser.add_argument(
        "--data", type=str, help="specify which dataset", default="tgbl-wiki"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="specify which model to run",
        default="qwen1.7b",
    )

    seed = 789001

    args = parser.parse_args()

    if args.model == "qwen1.7b":
        model_name = "Qwen/Qwen3-1.7B"
    elif args.model == "qwen4b":
        model_name = "Qwen/Qwen3-4B"
    elif args.model == "qwen8b":
        model_name = "Qwen/Qwen3-8B"
    elif args.model == "qwen14b":
        model_name = "Qwen/Qwen3-14B"
    elif args.model == "qwen2.5b":
        model_name = "Qwen/Qwen2.5-7B"
    elif args.model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model == "mistralv0.3":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    else:
        model_name = "Qwen/Qwen3-1.7B"

    print(f"Using model: {model_name}")

    ## Load up model
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda" if use_cuda else "cpu")
    model = models.transformers(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Setup generators
    generator = generate.json(model, schema_object=TGBExplanation)
    explanation_generator = generate.choice(model, TGBExplanationCat)

    # Load up system prompt
    with open("./prompts/explanation/system_prompt.txt", "r") as f:
        explanation_system_prompt = f.read()

    # Load up cache (prompt & data)
    prompt_cache = (
        f"./output/{args.data}/gpt-4.1-mini-2025-04-14/base_v1_icl/{args.data}*.jsonl"
    )
    print(f"Prompt cache: {glob(prompt_cache)}")
    messages_cache = []
    for file in glob(prompt_cache):
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                messages_cache.append(data["body"]["messages"])

    # Load up answer_key_cache
    print(f"Loading up answer_cache")
    with open(f"./answer_cache/{args.data}{args.model}_icl_cache.csv", "r") as f:
        answer_cache = pd.read_csv(f, header=None, names=["pred", "true"])
    print(f"Answer cache: {answer_cache.head()}")

    final_results = []
    batch_size = 100
    output_dir = f"./explanation/{args.data}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/explanation.jsonl"

    for i, prev_message in enumerate(tqdm(messages_cache, desc="Processing Messages")):

        result = {
            "pred": answer_cache.iloc[i]["pred"],
            "true": answer_cache.iloc[i]["true"],
            "explanation": "TBD",
            "category": "TBD",
        }

        messages = prev_message

        answer = answer_cache.iloc[i]["pred"]
        true_answer = answer_cache.iloc[i]["true"]

        # Get explanation
        messages = make_explanation_messages(messages, answer)
        explanation_model_prompt = make_model_prompt(tokenizer, messages)
        explanation_output = generator(explanation_model_prompt, seed=seed)
        explanation = explanation_output["explanation"]
        print(explanation)
        result["explanation"] = explanation

        # Get explanation category
        messages = make_explanation_category_messages(
            explanation_system_prompt, explanation
        )
        explanation_category_prompt = make_model_prompt(tokenizer, messages)
        explanation_category = explanation_generator(
            explanation_category_prompt, seed=seed
        )
        print(repr(explanation_category))
        result["category"] = explanation_category

        print(f"Result: {result}")

        final_results.append(result)

        # Batch save every 100
        if (i + 1) % batch_size == 0 or (i + 1) == len(messages_cache):
            with open(output_file, "a") as f:
                for r in final_results:
                    f.write(json.dumps(r, default=sanitize) + "\n")
            print(f"Appended {len(final_results)} results to {output_file}")
            final_results = []
