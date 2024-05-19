#!/bin/bash

methods=${1:-'GCoNet_plus,MCCL,GCoNet,DCFM,CADC,ICNet,MCCL,CoADNet,GCAGC'} && methods=(`echo ${methods} | tr ',' ' '`) && methods=${methods[@]}
data_lst=${2:-'random,CoCA'} && data_lst=(`echo ${data_lst} | tr ',' ' '`) && data_lst=${data_lst[@]}
repeat_indices=${3:-'1,2,3,4,5'} && repeat_indices=(`echo ${repeat_indices} | tr ',' ' '`) && repeat_indices=${repeat_indices[@]}

for data in ${data_lst}; do
    for method in ${methods}; do
        for idx_repeat in ${repeat_indices}; do
            CUDA_VISIBLE_DEVICES=0 python main.py --models ${method} --data ${data}
            rm -rf __pycache__ */__pycache__ */*/__pycache__
        done
    done
done
