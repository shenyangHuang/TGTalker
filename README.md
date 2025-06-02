# TGLLM
testing LLM on TG tasks

## script to use

- `tn_test.py` is the main scrip for running test evaluation
- `tn_val.py` is the script for running validation set (not that it is not as updated as `tn_test.py` at the moment)
- `json_test.py` attemps to use structured json output for LLM, don't use at the moment


## How to run scripts `reasoning_main.py`
```
#* base model
CUDA_VISIBLE_DEVICES=0 python -u reasoning_main.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-subreddit --nbr 2 

#* base model + icl
CUDA_VISIBLE_DEVICES=0 python -u reasoning_main.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-subreddit --nbr 2 --icl

#* base model + cot
CUDA_VISIBLE_DEVICES=0 python -u reasoning_main.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-subreddit --nbr 2 --cot --logfile reddit_log.json


#* base model + cot + icl
CUDA_VISIBLE_DEVICES=0 python -u reasoning_main.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-subreddit --nbr 2 --cot --icl --logfile reddit_log.json
```



## how to run scripts (old)
```
CUDA_VISIBLE_DEVICES=0 python -u tn_test.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-wiki --nbr 2 --icl
CUDA_VISIBLE_DEVICES=0 python -u tn_test.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-subreddit --nbr 2 --icl
CUDA_VISIBLE_DEVICES=0 python -u tn_test.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-lastfm --nbr 2 --icl
CUDA_VISIBLE_DEVICES=0 python -u tn_test.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-uci --nbr 2 --icl
CUDA_VISIBLE_DEVICES=0 python -u tn_test.py --batch 200 --model qwen1.7b --in_size 5 --bg_size 300 --data tgbl-enron --nbr 2 --icl
```

## Installation
```
module load python=3.10
python -m venv vllm_env
source vllm_env/bin/activate
pip install vllm
```

install TGB from source to download datasets and run evaluation
```
git clone git@github.com:shenyangHuang/TGB.git
pip install -e .   
```