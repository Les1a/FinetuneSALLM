CUDA_VISIBLE_DEVICES=4 python model/opt.py facebook/opt-1.3b --wbits 2 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=4 python model/opt.py facebook/opt-1.3b --wbits 3 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-1.3b --wbits 4 --groupsize -1 --lat --bcq_round 50 &
CUDA_VISIBLE_DEVICES=5 python model/opt.py facebook/opt-1.3b --wbits 8 --groupsize -1 --lat --bcq_round 50 &

wait


