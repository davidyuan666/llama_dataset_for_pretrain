我们可以添加gradient_checkpointing True参数进一步节省显存，增大per_device_train_batchsize。可以添加--sft_packing参数，该参数能够将多个样本拼接在一起，防止计算资源浪费。

```bash
CUDA_VISIBLE_DEVICES=0 python  src/train_bash.py \
      --stage sft \
      --model_name_or_path /workspace/sa/LLM_test/LLM_models/yuan2-2B/ \
      --do_train \
      --dataset alpaca_gpt4_zh,self_cognition,sharegpt_zh \
      --finetuning_type lora  \
      --lora_target q_proj,v_proj \
      --output_dir yuan2_2B_test_1gpu_qlora_checkpoint\
      --overwrite_cache \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 4  \
      --gradient_accumulation_steps 16  \
      --preprocessing_num_workers 16 \
      --lr_scheduler_type cosine \
      --logging_steps 10    \
      --save_steps 10000   \
      --learning_rate 5e-4   \
      --max_grad_norm 0.5     \
      --num_train_epochs 3   \
      --evaluation_strategy no  \
      --bf16 \
      --template yuan \
      --overwrite_output_dir    \
      --cutoff_len 1024\
      --quantization_bit 4 \
      --sft_packing
```
