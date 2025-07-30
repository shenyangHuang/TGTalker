import torch
import numpy as np
import argparse
from enum import Enum
from pydantic import BaseModel, constr
from transformers import AutoTokenizer
from outlines import models, generate, samplers
from utils import row2text, predict_link, set_seed, append2csv
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset
from neighbor_tracker import NeighborTracker
import json
import os
import timeit

"""
making the script pre-emp ready
Input Arguments
"""
parser = argparse.ArgumentParser('*** arguments for running LLM ***')
parser.add_argument('--seed', type=int, help='Random seed for inference', default=42)
parser.add_argument('--batch', type=int, help='Batch size', default=200)
parser.add_argument('--in_size', type=int, help='instruct set size', default=5)
parser.add_argument('--bg_size', type=int, help='background rows size', default=1000)
parser.add_argument('--model', type=str, help='specify which model to run', default='qwen1.5b')

parser.add_argument("--cot", action="store_true", default=False, help="now use chain of thought")
parser.add_argument("--cache_dst", action="store_true", default=False, help="caching destination node output")
parser.add_argument("--icl", action="store_true", default=False, help="now use icl")
parser.add_argument("--wandb", action="store_true", default=False, help="now using wandb")
parser.add_argument("--test", action="store_true", default=False, help="test data pipeline")
parser.add_argument('--data', type=str, help='specify which dataset', default='tgbl-wiki')
parser.add_argument('--nbr', type=int, help='how many neighbors to retrieve from the past', default=0)
parser.add_argument('--cot_size', type=int, help='how many cot to record until termination', default=5000)
parser.add_argument('--logfile', type=str, help='where to save the output', default=None)


args = parser.parse_args()
batch_size = args.batch #5 #200
instruct_size = args.in_size #5
background_size = args.bg_size #900
cot = args.cot
DATA = args.data
print ("using the following arguments: ")
print (args)


set_seed(args.seed)
# valid_cots = 0


"""
json template for reasoning
"""
class Step(BaseModel):
    explanation: str
    output: str


class TGBReasoning(BaseModel):
    steps: list[Step]
    destination_node: int


class TGBAnswer(BaseModel):
    destination_node: int


use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:0" if use_cuda else "cpu")

model_name = "Qwen/Qwen3-1.7B"

if (args.model == "qwen1.7b"):
    print ("using model Qwen/Qwen3-1.7B")
    model_name = "Qwen/Qwen3-1.7B"
elif (args.model == "qwen4b"):
    print ("using model Qwen/Qwen3-4B")
    model_name = "Qwen/Qwen3-4B"
elif (args.model == "qwen8b"):
    print ("using model Qwen/Qwen3-8B")
    model_name = "Qwen/Qwen3-8B"
elif (args.model == "qwen7b"):
    print ("using model Qwen/Qwen2.5-7B-Instruct")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
elif (args.model == "qwen14b"):
    print ("using model Qwen/Qwen3-14B")
    model_name = "Qwen/Qwen3-14B"
elif (args.model == "llama3"):
    print ("using meta-llama/Meta-Llama-3-8B-Instruct")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
elif (args.model == "mistralv0.3"):
    print ("using mistralai/Mistral-7B-Instruct-v0.3")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
else:
    print ("model not recognized, default to qwen 1.7b")
    model_name = "Qwen/Qwen3-1.7B"

if (args.cot):
    model = models.transformers(model_name, device=device)
else:
    model = models.vllm(
        model_name,
        dtype="half", 
        enable_prefix_caching=True,
        device=device,
    )


if (args.cot):
    generator = generate.json(model, TGBReasoning)
else:
    generator = generate.json(model, TGBAnswer)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Draw a sample
seed = args.seed

pred_name = ""

if (args.logfile is not None):
    array_name = args.logfile.split(".")[0] + ".npy"
    pred_name = args.logfile.split(".")[0]
else:
    array_name = args.data
    array_name += args.model
    if (args.icl):
        array_name += "_icl"
    if (args.cot):
        array_name += "_cot"
    if (args.cache_dst):
        array_name += "_cache"
    pred_name = array_name
    array_name += ".npy"

print (pred_name)

def make_prompt(system_prompt, src, ts, nbr_tracker=None):

    # add temporal neighbor information
    user_prompt = make_user_prompt(src, ts, nbr_tracker=nbr_tracker)
    prompt = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ],
    tokenize=False,
    add_bos=True,
    add_generation_prompt=True,
    )
    return prompt

def make_user_prompt(src, ts, nbr_tracker=None, hops=1):
    if (nbr_tracker is not None):
        user_prompt = (
            f",Source Node` {src} has the following past interactions:\n"
        )
        if (hops == 1):
            nbrs = nbr_tracker.get_neighbor([src])
            for _, v in nbrs.items():
                for destination, timestamp in v:
                    dst = int(destination)
                    timestamp = int(timestamp)
                    user_prompt += f"{src}, {dst}, {timestamp}) \n"
        else:
            seed_set = [src] 
            new_seeds = []
            for hop in range(hops):
                user_prompt += (f",Source Node` {src} has the following {hop}-hop interactions:\n")
                nbrs = nbr_tracker.get_neighbor(seed_set)
                for seed, v in nbrs.items():
                    for destination, timestamp in v:
                        dst = int(destination)
                        new_seeds.append(dst)
                        timestamp = int(timestamp)
                        user_prompt += f"{seed}, {dst}, {timestamp}) \n"
                seed_set = new_seeds
        user_prompt += f"Please predict the most likely `Destination Node` for `Source Node` {src} at `Timestamp` {ts}."
    else:
        user_prompt = (
            f"Predict the next interaction for source node {src} at time {ts},"
        )
    return user_prompt



def make_system_prompt(background_rows, instruct_strs, answer_strs, use_icl=True):
    system_prompt = (
            f"You are an expert temporal graph learning agent. Your task is to predict the next interaction (i.e. Destination Node) given the `Source Node` and `Timestamp`.\n\n" 
            f"Description of the temporal graph is provided below, where each line is a tuple of (`Source Node`, `Destination Node`, `Timestamp`).\n\nTEMPORAL GRAPH:\n"
        ) # system prompt
    if (background_size > 0):
        bg_str = row2text(background_rows)
        system_prompt += bg_str
    if (args.cot):
        system_prompt += "Let's think step by step about the problem.\n"
    if (use_icl):
        assert len(instruct_strs) == len(answer_strs), "instruct and answer strings must be the same cardinality"
        for i in range(len(instruct_strs)):
            system_prompt += instruct_strs[i]
            system_prompt += answer_strs[i]
    return system_prompt
        

def make_answer_prompt(dst):
    answer_prompt = ("{" + 
        f'"destination_node":{dst}' + "}"
    )
    return answer_prompt

def send_message(prompt):
    try:
        output = generator(prompt, seed=seed)
        dst_id = int(output.destination_node)
        if (args.cot):
            explanation = output.steps[0].explanation
        else:
            explanation = "Chain of Thought not used"
    except:
        dst_id = -1
        explanation = "Error in generating response"
    return dst_id, explanation



# #* load dataset from tgb
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



#* initialize neighbor tracker
idx = val_src.shape[0] - instruct_size -1
if (args.nbr > 0):
    tracker = NeighborTracker(val_src[0:idx],
                    val_dst[0:idx],
                    val_ts[0:idx],
                    max_size=args.nbr,)
else:
    tracker = None


if (background_size > 0):
    background_rows = val_rows[-(background_size + instruct_size +1):-(instruct_size+1)]
else:
    background_rows = []


instruct_strs = []
answer_strs = []
for i in range(instruct_size):
    if (args.nbr > 0):
        instruct_str = make_user_prompt(val_rows[idx+i][0], val_rows[idx+i][2], nbr_tracker=tracker)
    else:
        instruct_str = make_user_prompt(val_rows[idx+i][0], val_rows[idx+i][2])
    answer_str = make_answer_prompt(val_rows[idx+i][1])
    instruct_strs.append(instruct_str)
    answer_strs.append(answer_str)

if (args.nbr > 0):
    tracker.update(val_src[idx:],
                val_dst[idx:],
                val_ts[idx:])


#* initialization

if os.path.exists(array_name):
    scores = np.load(array_name)
    # Find indices where the value is zero
    zero_indices = np.where(scores == 0)[0]
    start_idx = zero_indices[0] if zero_indices.size > 0 else None
else:
    scores = np.zeros(test_rows.shape[0])
    start_idx = 0

tempt_scores = np.copy(scores)

cached_str = None
new_instructs = []
new_answers = []
val_ptr = idx
test_ptr = 0
split_mode = "test"
dataset.load_test_ns()

start_time = timeit.default_timer()
answer_json = []
error_prompts = 0


if (args.cache_dst):
    dst_out = []


for i in range(test_rows.shape[0]):
    #* model receives instruct on the batch that it processed
    if ((i%batch_size == 0) and (i>0)):

        if (i>= start_idx):
            scores = np.copy(tempt_scores)
            np.save(array_name, scores)    #* caching the numpy array 
            if (args.cache_dst):
                append2csv(pred_name + ".csv", dst_out)
                dst_out = []
            if (i == start_idx):
                print ("starting from ", start_idx)
                print ("average score so far is: ", np.mean(scores[0:i]), "for ", i, "edges")
            else:
                print ("-------------------------------")
                print ("batch number: ", i//batch_size)
                print ("average score so far is: ", np.mean(scores[0:i]), "for ", i, "edges")
                print ("the score of current batch is", np.mean(scores[(i-batch_size):i]))
                print ("it takes ", timeit.default_timer()-start_time, " seconds to process ", i, " edges")
                print ("-------------------------------")

        cached_str = None  # empty cache

        #* for background, move into the instruct set of the training set first then into validation set
        if (background_size > 0):
            background_rows = background_rows[batch_size:]
            if (val_ptr >= val_rows.shape[0]):
                assert ((test_ptr+batch_size)<= i)
                new_rows = test_rows[test_ptr:test_ptr+batch_size]
                test_ptr += batch_size
            else:
                assert ((test_ptr+batch_size)<= i)
                val_end = min(val_rows.shape[0]-1, val_ptr+batch_size)
                new_rows = val_rows[val_ptr:val_end]
                if (val_end - val_ptr) < batch_size:
                    remain = batch_size - (val_end - val_ptr)
                    new_rows = np.concatenate((new_rows,test_rows[test_ptr:test_ptr+remain]), axis=0)
                    test_ptr += remain
                val_ptr += batch_size


            background_rows = np.concatenate((background_rows, new_rows), axis=0) 
            assert background_rows.shape[0] == background_size, "background size not correct"

        instruct_strs = instruct_strs[batch_size:]
        instruct_strs = np.concatenate((instruct_strs, new_instructs[-instruct_size:]), axis=0)
        assert instruct_strs.shape[0] == instruct_size, "instruct size not correct"     
        answer_strs = answer_strs[batch_size:]
        answer_strs = np.concatenate((answer_strs, new_answers[-instruct_size:]), axis=0)        
        assert answer_strs.shape[0] == instruct_size, "answer size not correct"
        
        
        new_instructs = []
        new_answers = []

        if (args.nbr > 0):
            tracker.update(test_src[(i-batch_size):i], test_dst[(i-batch_size):i], test_ts[(i-batch_size):i])

    instruct_str = make_user_prompt(test_rows[i][0], test_rows[i][2], nbr_tracker=tracker)
    answer_str = make_answer_prompt(test_rows[i][1])
    if (cached_str is None):
        history = make_system_prompt(background_rows, instruct_strs, answer_strs, use_icl=args.icl)
    else:
        history = cached_str

    # append for processing after the batch
    new_instructs.append(instruct_str)
    new_answers.append(answer_str)

    if (i >= start_idx):
        predicted_dst, reasoning = send_message(make_prompt(history,test_rows[i][0], test_rows[i][2], nbr_tracker=tracker))
        if (args.cache_dst):
            dst_out.append([predicted_dst, test_rows[i][1]])
        if (reasoning != "Error in generating response"):
            # valid_cots += 1
            answer_json = [{"destination": predicted_dst, "true_answer":int(test_rows[i][1]) ,"explanation": reasoning}]
        else:
            answer_json = []
            error_prompts += 1
        
        if (args.logfile is not None and len(answer_json) > 0):
            with open(args.logfile, 'a') as f:
                json.dump(answer_json, f, indent=4) 
                f.write('\n')  

        neg_batch_list = neg_sampler.query_batch(np.array([test_rows[i][0]]), np.array([int(test_rows[i][1])]), np.array([int(test_rows[i][2])]), split_mode=split_mode)
        for _, neg_batch in enumerate(neg_batch_list):    
            query_dst = np.concatenate([np.array([int(test_rows[i][1])]), neg_batch])
            y_pred = predict_link(query_dst, predicted_dst)
            input_dict = {
                    "y_pred_pos": np.array([y_pred[0]]),
                    "y_pred_neg": np.array(y_pred[1:]),
                    "eval_metric": [metric],
                }
            tempt_scores[i] = evaluator.eval(input_dict)[metric]
        


scores = np.copy(tempt_scores)
if (args.cache_dst):
    append2csv(pred_name + ".csv", dst_out)
print(f"Final average score: {np.mean(scores):.4f} over {scores.shape[0]} edges")
print("there are ", error_prompts, " prompts that were parsed with error")



