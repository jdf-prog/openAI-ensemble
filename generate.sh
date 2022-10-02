job_id=$(python get_job_id.py)
data_path="./data/cnn_dailymail_test_hypo_min.jsonl"
model="text-curie-001" # "text-curie-001", "text-davinci-002"
output_path="./output/${model}/test_output${job_id}.jsonl"
job_record="./jobs/${job_id}.txt"
mkdir -p "./jobs"
echo "submitted to job ${job_id}"
echo "generate.sh" > ${job_record}
echo "job_id: ${job_id}" >> ${job_record}
echo "model: ${model}" >> ${job_record}
echo "data_path: ${data_path}" >> ${job_record}
echo "output_path: ${output_path}" >> ${job_record}

python src/generate.py \
    --model ${model} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --num_hypos 6 \
    --temperature 0.6 \
    --max_tokens 150 \
    --overwrite \
    --num_few_shot 0 \
    >> ${job_record} 2>&1 \
