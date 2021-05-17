#!/bin/sh
#SBATCH -J jtvae
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p p-RTX2080

# python ../fast_jtnn/mol_tree.py -i ./../data/zinc/train.txt -v ./../data/zinc/vocab.txt
python -u vae_train.py --train zinc-processed --vocab ../data/zinc/vocab.txt --save_dir zinc_vae_model/
python -u sample.py --nsample 100 --vocab ../data/zinc/vocab.txt --hidden 450 --model zinc_vae_model/model.epoch-19 --output_file './zinc_sample.txt'
