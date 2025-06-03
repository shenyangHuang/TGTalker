"""
Generate the batch file to be processed for temporal graph learning with LLM
"""
import re
import numpy as np
import pandas as pd
import argparse
import json

from pprint import pprint
from functools import partial

from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset


from utils import load_txt_from_edgelist, load_rows_from_edgelist, row2text
from neighbor_tracker import NeighborTracker
from gpt_batch import generate_messages, write_chunked_files



def create_system_message(system_prompt: str, 
                          background_rows: list[str], 
                          cot: bool) -> str:
    """
    Create the system message.
    """
    out = system_prompt
    out += "\n\nTEMPORAL GRAPH:\n"

    for row in background_rows:
        out += f"({row[0]}, {row[1]}, {row[2]})\n"
    
    if cot:
        out += "\nLet's think step by step about the problem.\n"

    return out


def create_user_message(row: list, neighbors) -> str:
    """
    Create the user_message.

    A row within the training dataset.

    """

    if (args.nbr == 0):
        out =  f"`Source Node` {row[0]} has no past interactions:\n"
    else:
        out = f"`Source Node` {row[0]} has the following past interactions:\n"
        for src, v in neighbors.items():
            for dst, ts in v:
                dst = int(dst)
                ts = int(ts)
                out += f"({src}, {dst}, {ts})\n"

    out += f"Please predict the most likely `Destination Node` for `Source Node` {row[0]} at `Timestamp` {row[2]}."
    return out


def create_assistant_message(row: list) -> str:
    """
    Create the assistant message.

    A row within the training dataset.
    """
    answer_prompt = ("{" + 
        f'"destination_node":{row[1]}' + "}"
    )
    return answer_prompt


def create_input(
        id: int, 
        model: str,
        messages: list,
        dataset: str,
        json_schema: str,
        temperature: float
        ) -> dict:
    """
    Create the inputs for the model.
    """
    return {
        "custom_id": f"{dataset}-{id}",
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

    parser.add_argument('--batch', type=int, help='Batch size', default=200)
    parser.add_argument('--in_size', type=int, help='instruct set size', default=5)
    parser.add_argument('--bg_size', type=int, help='background rows size', default=1000)
    parser.add_argument('--model', type=str, help='specify which model to run', default='qwen1.5b')
    parser.add_argument("--cot", action="store_true", default=False, help="Use chain of thought or not")
    parser.add_argument("--icl", action="store_true", default=False, help="Use in context learning or not")
    parser.add_argument("--test", action="store_true", default=False, help="Test Data Pipeline")
    parser.add_argument('--data', type=str, help='specify which dataset', default='tgbl-wiki')
    parser.add_argument('--nbr', type=int, help='how many neighbors to retrieve from the past', default=0)
    parser.add_argument('--logfile', type=str, help='where to save the output', default='log.json')
    parser.add_argument('--prompt_version', type=str, help='version of prompt to use', default='v1')
    parser.add_argument('--output_dir', type=str, help='where to save the batch file', default='./output')
    parser.add_argument('--temperature', type=float, help='temperature to run the model', default=1.0)
    parser.add_argument('--max_no_of_prompts', type=int, help='Max number of batch prompts', default=-1)

    args = parser.parse_args()

    with open(f"./prompts/{args.prompt_version}/system_prompt.txt") as f:
        system_prompt = f.read()

    schema_file_name = "TGBReasoning_schema.json" if args.cot else "TGBAnswer_schema.json"
    with open(f"./prompts/{args.prompt_version}/{schema_file_name}") as f:
        json_schema = json.load(f)
    
    batch_size = args.batch #5 #200
    instruct_size = args.in_size #5
    background_size = args.bg_size #900
    cot = args.cot
    DATA = args.data

    prompt_type = f"base_{args.prompt_version}"
    if args.cot:
        prompt_type += "_cot"
    if args.icl:
        prompt_type += "_icl"

    result_folder = f"{args.output_dir}/{args.data}/{args.model}/{prompt_type}"
    answer_key = f"{result_folder}/answer_key.parquet"
    batch_dir = f"{result_folder}/"

    # #* load dataset from tgb
    # data loading with `numpy`
    dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
    data = dataset.full_data  
    metric = dataset.eval_metric
    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler


    # get masks
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    val_src = np.concatenate([data['sources'][val_mask]])
    val_dst = np.concatenate([data['destinations'][val_mask]])
    val_ts = np.concatenate([data['timestamps'][val_mask]])
    val_rows = np.stack((val_src, val_dst, val_ts), axis=1)

    test_src = np.concatenate([data['sources'][test_mask]])
    test_dst = np.concatenate([data['destinations'][test_mask]])
    test_ts = np.concatenate([data['timestamps'][test_mask]])
    test_rows = np.stack((test_src, test_dst, test_ts), axis=1)

    #! initialize neighbor tracker
    idx = val_src.shape[0] - instruct_size -1
    if (args.nbr > 0):
        tracker = NeighborTracker(val_src[0:idx],
                        val_dst[0:idx],
                        val_ts[0:idx],
                        max_size=args.nbr,)
    background_rows = val_rows[-(background_size + instruct_size +1):-(instruct_size+1)]

    
    usr_msgs = []
    assistant_msgs = []

    for i in range(instruct_size):

        neighbors = None
        if (args.nbr > 0):
            neighbors = tracker.get_neighbor([val_rows[idx+i][0]])

        usr_msg  = create_user_message(val_rows[idx+i], neighbors)
        assistant_msg = create_assistant_message(val_rows[idx+i])
        
        usr_msgs.append(usr_msg)
        assistant_msgs.append(assistant_msg)

    if (args.nbr > 0):
        tracker.update(val_src[idx:],
                       val_dst[idx:],
                       val_ts[idx:])


    cached_str = None
    new_usr_msgs = []
    new_assistant_msgs = []

    val_ptr = idx
    test_ptr = 0

    batch_inputs = []
    batch_answers = []
    batch_query_dst = []

    split_mode = "test"
    dataset.load_test_ns()


    # on the validation set
    for i in range(test_rows.shape[0]):

        if ((i%batch_size == 0) and (i>0)):
            cached_str = None 
            #! move forward the prompt in chronological order
            #* for background, move into the instruct set of the training set first then into validation set
            background_rows = background_rows[batch_size:]
            if (val_ptr >= val_rows.shape[0]):
                new_rows = test_rows[test_ptr:test_ptr+batch_size]
                test_ptr += batch_size
            else:
                val_end = min(val_rows.shape[0]-1, val_ptr+batch_size)
                new_rows = val_rows[val_ptr:val_end]
                if (val_end - val_ptr) < batch_size:
                    remain = batch_size - (val_end - val_ptr)
                    new_rows = np.concatenate((new_rows,test_rows[test_ptr:test_ptr+remain]), axis=0)
                    test_ptr += remain
                val_ptr += batch_size


            background_rows = np.concatenate((background_rows, new_rows), axis=0)     

            assert background_rows.shape[0] == background_size, "background size not correct"

            usr_msgs = usr_msgs[batch_size:]
            usr_msgs = np.concatenate((usr_msgs, new_usr_msgs[-instruct_size:]), axis=0)
            assert usr_msgs.shape[0] == instruct_size, "instruct size not correct"

            assistant_msgs = assistant_msgs[batch_size:]
            assistant_msgs = np.concatenate((assistant_msgs, new_assistant_msgs[-instruct_size:]), axis=0)  
            assert assistant_msgs.shape[0] == instruct_size, "answer size not correct"
            
            
            new_usr_msgs = []
            new_assistant_msgs = []

            if (args.nbr > 0):
                tracker.update(test_src[(i-batch_size):i], test_dst[(i-batch_size):i], test_ts[(i-batch_size):i])

        
        final_usr_msg = create_user_message(test_rows[i], neighbors)
        final_assistant_msg = create_assistant_message(test_rows[i])

        if (cached_str is None):
            final_system_prompt = create_system_message(system_prompt, 
                                            background_rows, 
                                            cot)
        else:
            final_system_prompt = cached_str

        # append for processing after the batch
        new_usr_msgs.append(final_usr_msg)
        new_assistant_msgs.append(final_assistant_msg)


        # Generate Prompt, enable ICL
        chat_messages=[]

        if (args.icl):
            if len(usr_msgs) == len(assistant_msgs):
                for user_msg, assistant_msg in zip(usr_msgs, assistant_msgs):
                    chat_messages.append(user_msg)
                    chat_messages.append(assistant_msg)

        chat_messages.append(final_usr_msg)

        model_input =  generate_messages(system_prompt=final_system_prompt,
                                        messages=chat_messages,
                                )
            
        
        batch_inputs.append(model_input)
        batch_answers.append(final_assistant_msg)

        if args.test:
            pprint(model_input)

        neg_batch_list = neg_sampler.query_batch(np.array([test_rows[i][0]]), np.array([int(test_rows[i][1])]), np.array([int(test_rows[i][2])]), split_mode=split_mode)
        query_dsts = []
        for _, neg_batch in enumerate(neg_batch_list):    
            query_dst = np.concatenate([np.array([int(test_rows[i][1])]), neg_batch])
            query_dsts.append(query_dst)
            
            if args.test:
                print(f"Query_dst: {query_dst}")
        batch_query_dst.append(query_dsts)
        
        if (args.test) and (i > 1):
            break

        if (args.max_no_of_prompts > 0) and (len(batch_inputs) >= args.max_no_of_prompts):
            break
    

    create_input_specified = partial(create_input, 
                                    dataset=DATA,
                                    json_schema=json_schema,
                                    temperature=args.temperature)

    # Save Batch Prompts
    write_chunked_files(
        data_iterator=batch_inputs,
        folder=batch_dir,
        base_filename=f"{args.data}.jsonl",
        model=args.model,
        output_token_estimate = 200,
        output_token_conservative_estimate = 415,
        chunk_size=10000, #5000 (ICL)   10000 (Base / COT)     Max: 50,000, 200 MB of file size
        create_input_func=create_input_specified
    )
    
    # Save Answer Key
    df = pd.DataFrame(batch_answers, columns=["answer"])
    df["query_dst"] = batch_query_dst
    df["task_id"] = [f"{args.data}-{i}" for i in range(len(batch_answers))]
    df.to_parquet(answer_key, index=False)
        
        

