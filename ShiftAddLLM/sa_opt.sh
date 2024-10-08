
CUDA_VISIBLE_DEVICES=1 python model/opt.py facebook/opt-125m --wbits 3 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=1 python model/opt.py facebook/opt-125m --wbits 2 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=1 python model/opt.py facebook/opt-125m --wbits 4 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=1 python model/opt.py facebook/opt-125m --wbits 8 --groupsize -1 --acc --bcq_round 20 &


CUDA_VISIBLE_DEVICES=1 python model/opt.py facebook/opt-350m --wbits 3 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=2 python model/opt.py facebook/opt-350m --wbits 2 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=2 python model/opt.py facebook/opt-350m --wbits 4 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=2 python model/opt.py facebook/opt-350m --wbits 8 --groupsize -1 --acc --bcq_round 20 &


CUDA_VISIBLE_DEVICES=3 python model/opt.py facebook/opt-1.3b --wbits 3 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=3 python model/opt.py facebook/opt-1.3b --wbits 2 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=4 python model/opt.py facebook/opt-1.3b --wbits 4 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=4 python model/opt.py facebook/opt-1.3b --wbits 8 --groupsize -1 --acc --bcq_round 20 &


CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-125m --wbits 2 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-125m --wbits 3 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-125m --wbits 4 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-125m --wbits 8 --groupsize -1 --lat --bcq_round 20 &


CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-350m --wbits 2 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-350m --wbits 3 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-350m --wbits 4 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-350m --wbits 8 --groupsize -1 --lat --bcq_round 20 &

CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-1.3b --wbits 2 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-1.3b --wbits 3 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=0 python model/opt.py facebook/opt-1.3b --wbits 4 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=0 python model/opt.py facebook/opt-1.3b --wbits 8 --groupsize -1 --lat --bcq_round 20 &

wait

cd ../llm-fintune
bash train.sh