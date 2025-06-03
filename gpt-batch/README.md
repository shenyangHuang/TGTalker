# OpenAI Batch Jobs

We assume that you already have an OpenAI account & appropriate key.

**Please save it in a local .env or environment variable.**

## Batch File Generation

We create the batch files to be submitted to OpenAI through the `generate_reasoning_main.py`.

```python
python ./generate_reasoning_main.py --batch 100 --model gpt-4.1-mini-2025-04-14 --in_size 5 --bg_size 300 --data tgbl-subreddit --nbr 2 --icl --max_no_of_prompts 5000
```

Our `gpt_batch` file currently supports the models: `["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "o4-mini-2025-04-16", "o3-2025-04-16"]`

This will automatically create the folders & output folders needed.

## Batch File w/ OpenAI

Use the `openai_batch.ipynb` to perform the actions: `Upload`, `Job Status (File Progress Check)`, or `Cancel` and finally `Download`.

Notes:

1. **Please edit the first cell `Parameter` to choose the right dataset.**
2. Please _**DO NOT RUN**_ the notebook as is. Run the cell one by one, based on the action needed.

## Scoring

Run the `openai_scorer.ipynb` to score the batch files. This will align the batch job id with the answer key saved before. From there, you'll receive the score for each dataset.