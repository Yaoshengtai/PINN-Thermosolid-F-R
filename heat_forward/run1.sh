#!/bin/bash
#SBATCH --gpus=1    
#参数在脚本中可以加上前缀“#SBATCH”指定，和在命令参数中指定功能一致，如果脚本中的参数和命令指定的参数冲突，则命令中指定的参数优先级更高。在此处指定后可以直接sbatch ./run.sh 提交。
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境
module load anaconda/2022.10
module load cuda/12.1
source activate ystpinn
 
#python程序运行，需在.py文件指定调用GPU，并设置合适的线程数，batch_size大小等
python heat_forward.py \
    --lr 0.000003 \
    --batch_size 4096 \
    --epochs 1000000 \
    --gpu True \
    --train_rec_size 512 \
    --train_bound_size 64 \
    --train_gen_random True \
    --valid_gen_random True \
    --weight_up 10 \
    --weight_left 10 \
    --weight_right 3 \
    --weight_bottom 1 \
    --boundary_strictness 20 \
    --network_MLP "(32,32,32,32,32)" \
    --check_every 250 \
    --save_dict "run1"\
    --maxf 10 \
    --impose 1 \
    --mtl 0





                    
