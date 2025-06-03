"""
Generate the batch file to classify the explanation for the temporal graph link prediction task.
"""
import re
import numpy as np
import pandas as pd
import argparse
import json

from pprint import pprint
from functools import partial

from gpt_batch import generate_messages, write_chunked_files


def create_input(
        id: int, 
        model: str,
        messages: list,
        custom_id_prepend: str,
        json_schema: str,
        temperature: float
        ) -> dict:
    """
    Create the inputs for the model.
    """
    return {
        "custom_id": f"{custom_id_prepend}-{id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model" : model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "TGB",
                    "schema": json_schema,
                    "strict": True
                }
            },
            "temperature": temperature
        }
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser('*** Arguments to generate LLM batch file ***')

    parser.add_argument('--data', type=str, help='dataset to use', default='0') 
    parser.add_argument('--model', type=str, help='specify which model to run', required=True)
    parser.add_argument('--prompt_version', type=str, help='version of prompt to use', default='explanation')

    args = parser.parse_args()

    with open(f"./prompts/{args.prompt_version}/system_prompt.txt") as f:
        system_prompt = f.read()

    schema_file_name = "explanation_schema.json"
    with open(f"./prompts/{args.prompt_version}/{schema_file_name}") as f:
        json_schema = json.load(f)
    

    df = pd.read_csv(f"./explanations/{args.model}/explanations_{args.data}.csv")

    batch_dir = f"./output/explanations/{args.model}/{args.data}"
    


    df["explanations"] = df["explanations"].apply(lambda x: 
                                                 generate_messages(system_prompt=system_prompt,
                                                 messages = [x])
                            )
    
    create_input_specified = partial(create_input, 
                                    custom_id_prepend = f"{args.data}-explanation",
                                    json_schema=json_schema,
                                    temperature=1.0)
            
    # Save Batch Prompts
    write_chunked_files(
        data_iterator=df["explanations"],
        folder=batch_dir,
        base_filename=f"{args.data}.jsonl",
        model=args.model,
        chunk_size=10000, #5000 (ICL)   10000 (Base / COT)     Max: 50,000, 200 MB of file size
        create_input_func=create_input_specified
    )
        

