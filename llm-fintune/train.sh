CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_16bits-acc.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_16bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=1 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_8bits-acc.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_8bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_4bits-acc.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_4bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_3bits-acc.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_3bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=6 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_2bits-acc.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_2bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait



CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_8bits-acc.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_8bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_4bits-acc.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_4bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_3bits-acc.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_3bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_2bits-acc.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_2bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_16bits-acc.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_16bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait



CUDA_VISIBLE_DEVICES=6,7 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_8bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_8bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=6,7 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_4bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_4bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=6,7 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_3bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_3bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &



CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-125m_2bits-lat.pth" \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-125m_2bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait



CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_8bits-lat.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_8bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_4bits-lat.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_4bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_3bits-lat.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_3bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-350m_2bits-lat.pth" \
    --model_name_or_path facebook/opt-350m \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=2.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-350m_2bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &





CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_8bits-acc.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_8bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=6 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_4bits-acc.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_4bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait


CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_3bits-acc.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_3bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &


CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_2bits-acc.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_2bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_16bits-acc.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_16bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &



CUDA_VISIBLE_DEVICES=6 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_8bits-lat.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_8bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_4bits-lat.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_4bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &



CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_3bits-lat.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_3bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-1.3b_2bits-lat.pth" \
    --model_name_or_path facebook/opt-1.3b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1.5e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-1.3b_2bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait


CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_2bits-lat.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_2bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &




CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-6.7b_3bits-lat.pth" \
    --model_name_or_path facebook/opt-6.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=5e-5 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-6.7b_3bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-6.7b_2bits-lat.pth" \
    --model_name_or_path facebook/opt-6.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=5e-5 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-6.7b_2bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-6.7b_3bits-acc.pth" \
    --model_name_or_path facebook/opt-6.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=5e-5 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-6.7b_3bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=6 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-6.7b_2bits-acc.pth" \
    --model_name_or_path facebook/opt-6.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=5e-5 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-6.7b_2bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &






CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_8bits-acc.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_8bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_4bits-acc.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_4bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_3bits-acc.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_3bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_2bits-acc.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_2bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_16bits-acc.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_16bits-acc-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &



CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_8bits-lat.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_8bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=6 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_4bits-lat.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_4bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

CUDA_VISIBLE_DEVICES=7 python train.py \
    --dataset_name="tatsu-lab/alpaca" --dataset_config_name='' --load_weight "../ShiftAddLLM/outputs/facebook/opt-2.7b_3bits-lat.pth" \
    --model_name_or_path facebook/opt-2.7b \
    --num_train_epochs=3 --block_size=2048 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=1 --optim paged_adamw_32bit \
    --learning_rate=1e-4 --torch_dtype bfloat16 --bf16 --output_dir='./output/opt-2.7b_3bits-lat-alpaca' --do_train --do_eval \
    --evaluation_strategy='epoch' --save_strategy='epoch' --logging_strategy='steps' \
    --logging_first_step --logging_steps=20 --save_total_limit=0 --run_name='opt-alpaca' --overwrite_output_dir --low_cpu_mem_usage \
    --bits 16 --lora_r 16 --lora_dropout 0.01 &

wait




