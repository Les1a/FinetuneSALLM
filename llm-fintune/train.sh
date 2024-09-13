# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_16bits.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_16bits-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=1 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_8bits.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_8bits-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_4bits.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_4bits-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_3bits.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_3bits-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_2bits.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_2bits-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01



CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_8bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_8bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_4bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_4bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_3bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_3bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_2bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_2bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01




# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_8bits-lat.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_8bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_4bits-lat.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_4bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_3bits-lat.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_3bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_2bits-lat.pth" \
#     --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
#     --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_2bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
#     --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
#     --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
#     --bits 16 --lora_r 16 --lora_dropout 0.01



CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_8bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_8bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_4bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_4bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_3bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_3bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_2bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_2bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01





CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_8bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_8bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_4bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_4bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_3bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_3bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_2bits.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_2bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01



CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_8bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_8bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_4bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_4bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_3bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_3bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_2bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_2bits-lat-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01







CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --model_name_or_path "ShiftAddLLM/opt6.7b-2bit-lat" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-6.7b_3bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="wikitext" --dataset_config_name='wikitext-2-raw-v1' --model_name_or_path "ShiftAddLLM/opt6.7b-2bit-lat" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=8 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-6.7b_2bits-wikitext-2-raw-v1' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01

