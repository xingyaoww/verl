export CUDA_VISIBLE_DEVICES=0,1
# Tested with 2 & 4 GPUs

nproc_per_node=2
save_path=./output

# Shift the arguments so $@ refers to the rest
shift 2
# NOTE:
# Micro_batch_size is in fact global batch size
# each gpu handles  micro_batch_size / #gpu data points
# sequence_parallism eaches this number to be < 1, to be concrete, be 1/ulysses_sequence_parallel_size
# We only validated the ulysses_sequence_parallel_size == #GPU case

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.micro_batch_size=1 \
    optim.lr=1e-4 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.total_epochs=5 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    +ulysses_sequence_parallel_size=2 \
    +use_remove_padding=true \
    +debug=false