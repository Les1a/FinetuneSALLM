
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-2.7b --wbits 3 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-2.7b --wbits 2 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-2.7b --wbits 4 --groupsize -1 --acc --bcq_round 20 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-2.7b --wbits 8 --groupsize -1 --acc --bcq_round 20 &

CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-2.7b --wbits 2 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=6 python model/opt.py facebook/opt-2.7b --wbits 3 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-2.7b --wbits 4 --groupsize -1 --lat --bcq_round 20 &
CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-2.7b --wbits 8 --groupsize -1 --lat --bcq_round 20 &

CUDA_VISIBLE_DEVICES=7 python model/opt.py facebook/opt-2.7b --wbits 16 --groupsize -1 --acc --bcq_round 20 &
