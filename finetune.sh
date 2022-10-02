base_model="curie" #  (ada, babbage, curie, or davinci).
train_file="./data/open_ai_train.jsonl"

echo "finetune.sh"
echo "base_model: ${base_model}"
echo "train_file: ${train_file}"

openai api fine_tunes.create \
    -t ${train_file} \
    -m ${base_model} \
    --suffix base \
    --n_epochs 2 \
