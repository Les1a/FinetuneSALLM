CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-125m --wbits 2 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-125m --wbits 3 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=3 python model/opt.py facebook/opt-125m --wbits 4 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=3 python model/opt.py facebook/opt-125m --wbits 8 --groupsize -1 --lat --bcq_round 50 &


CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-350m --wbits 2 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-350m --wbits 3 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-350m --wbits 4 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=3 python model/opt.py facebook/opt-350m --wbits 8 --groupsize -1 --lat --bcq_round 50 &