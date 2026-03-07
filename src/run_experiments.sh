#!/bin/bash

# # 2.4 - Vanishing Gradient Analysis
# echo "Running 2.4 experiments..."

# python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 2 -sz 128 128 -a sigmoid -w_i xavier -w_p da6401_a1
# python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a sigmoid -w_i xavier -w_p da6401_a1
# python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 4 -sz 128 128 128 128 -a sigmoid -w_i xavier -w_p da6401_a1

# python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 2 -sz 128 128 -a relu -w_i xavier -w_p da6401_a1
# python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a relu -w_i xavier -w_p da6401_a1
# python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 4 -sz 128 128 128 128 -a relu -w_i xavier -w_p da6401_a1

# echo "2.4 done!"

# 2.5 - Dead Neuron Investigation
python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.1 -nhl 3 -sz 128 128 128 -a relu -w_i xavier -w_p da6401_a1
python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.1 -nhl 3 -sz 128 128 128 -a tanh -w_i xavier -w_p da6401_a1

echo "2.5 done!"

#2.6 
echo "Running 2.6 experiments..."

python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy -w_p da6401_a1
python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l mse -w_p da6401_a1

echo "2.6 done!"