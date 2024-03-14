#!/bin/bash
#SBATCH --gpus=1    
#参数在脚本中可以加上前缀“#SBATCH”指定，和在命令参数中指定功能一致，如果脚本中的参数和命令指定的参数冲突，则命令中指定的参数优先级更高。在此处指定后可以直接sbatch ./run.sh 提交。
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境
module load anaconda/2022.10
module load cuda/12.1
source activate ystpinn
 
#python程序运行，需在.py文件指定调用GPU，并设置合适的线程数，batch_size大小等
python force_forward.py \
    --lr 0.001 \
    --batch_size 1024 \
    --epochs 1000000 \
    --gpu True \
    --train_rec_size 128 \
    --train_bound_size 128 \
    --train_gen_random True \
    --valid_gen_random True \
    --weight_up 20 \
    --weight_left 10 \
    --weight_right 5 \
    --weight_bottom 2 \
    --weight_equ1 5 \
    --weight_equ2 1 \
    --boundary_strictness 0.5 \
    --network_MLP "(128,128,128,128,128)" \
    --check_every 1000 \
    --save_dict "run5"\
    --maxf 10 \
    --impose 1 \
    --mtl 1




                    
