import openai
import os
import argparse
import json
import random
import logging
import prettytable as pt
import numpy as np
from evaluate import load as load_metric
from pathlib import Path
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_completion_data(item, num_hypos):
    hypos = [hypo for hypo in list(item['hypotheses'].values())[:num_hypos]]
    passage = item['original']
    prompt = ""
    prompt += f"Passage: \n{passage}\n"
    for i, hypo in enumerate(hypos):
        prompt += f"Summary {i}:\n{hypo['content']}\n"
    prompt += "Combined Summary:\n"

    completion = item['reference']
    return prompt, completion

def evaluate(completions, items):
    logger.info("Evaluating summaries")
    table = pt.PrettyTable(field_names=["source", "rouge1", 'rouge2', 'rougeL'])
    rouge = load_metric('rouge')
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for completion, item in zip(completions, items):
        reference = item['reference']
        generated_texts = [choice['text'] for choice in completion['choices']]
        scores = []
        for text in generated_texts:
            scores.append(rouge.compute(predictions=[text], references=[reference], rouge_types=['rouge1', 'rouge2', 'rougeL']))
        rouge1_scores.append(np.mean([score['rouge1'] for score in scores]))
        rouge2_scores.append(np.mean([score['rouge2'] for score in scores]))
        rougeL_scores.append(np.mean([score['rougeL'] for score in scores]))

    # record the performance in the table
    table.add_row([
        "gpt-3",
        round(np.mean(rouge1_scores), 3),
        round(np.mean(rouge2_scores), 3),
        round(np.mean(rougeL_scores), 3)
    ])
    sources = [source for source in items[0]['hypotheses']]
    for source in sources:
        rouge1_score = np.mean([item['hypotheses'][source]['metrics']['rouge1'] for item in items])
        rouge2_score = np.mean([item['hypotheses'][source]['metrics']['rouge2'] for item in items])
        rougeL_score = np.mean([item['hypotheses'][source]['metrics']['rougeL'] for item in items])
        table.add_row([source, rouge1_score, rouge2_score, rougeL_score])

    logger.info("Evaluation Results \n{}".format(table))
    result = {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
    }
    return result


def openai_generate(args):
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    if not data_path.exists():
        raise ValueError("data path does not exist")
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # debug
    with open("./data/cnn_format_items.jsonl", 'r') as f:
        format_data = [json.loads(line) for line in f]
        max_length = max([len(item['original']) for item in format_data])
        def get_mean_rouge(item):
            return np.mean([hypo['metrics']['rouge2'] for hypo in list(item['hypotheses'].values())])
        format_data = sorted(format_data, key=lambda x: len(x['original'])/max_length * 2 - get_mean_rouge(x) if get_mean_rouge(x) < 0.8 else 0)
    format_prompt = ""
    for i in range(args.num_few_shot):
        item=format_data[i]
        prompt, completion = get_completion_data(item, args.num_hypos)
        format_prompt += prompt
        format_prompt += completion
    format_prompt += "\n"

    results = []
    if not output_path.exists() or args.overwrite:
        logger.info("Generating summaries")
        for item in data:
            prompt, _ = get_completion_data(item, args.num_hypos)
            prompt = format_prompt + prompt
            completion = openai.Completion.create(
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
            )
            results.append({"prompt": prompt, "completion": completion})
    else:
        logger.info("Loading cached summaries")
        with open(output_path, 'r') as f:
            results = [json.loads(line) for line in f]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    completions = [result['completion'] for result in results]
    evaluate(completions, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="curie")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_hypos", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--num_few_shot", type=int, default=0, help="number of few-shot examples to use")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    for key, value in vars(args).items():
        logger.info("{}: {}".format(key, value))
    openai_generate(args)
