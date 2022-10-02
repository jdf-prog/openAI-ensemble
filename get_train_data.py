import json
import os
import argparse
from src.generate import get_completion_data
from pathlib import Path

def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise ValueError("data path does not exist")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    result = []
    for item in data:
        prompt, completion = get_completion_data(item, args.num_hypos)
        result.append({
            "prompt": prompt,
            "completion": completion
            })

    with open(output_path, 'w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_hypos", type=int, default=-1)
    args = parser.parse_args()
    main(args)

