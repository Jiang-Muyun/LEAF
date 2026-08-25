python init_text_embeddings.py all-mpnet-base-v2
python init_text_embeddings.py qwen3-embedding-4B

python a_pretrain.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1

# Instruction-tuning examples using relative pretraining-checkpoint paths.
# Epoch 5 example (uncomment to run):
# python b_instruct_tuning.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1 --seed 0 --pretrain_ckpt checkpoints/leaf-pretrain-epoch-05.ckpt

# Epoch 10 examples with three random seeds:
python b_instruct_tuning.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1 --seed 0 --pretrain_ckpt checkpoints/leaf-pretrain-epoch-05.ckpt
python b_instruct_tuning.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1 --seed 0 --pretrain_ckpt checkpoints/leaf-pretrain-epoch-10.ckpt
python b_instruct_tuning.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1 --seed 1 --pretrain_ckpt checkpoints/leaf-pretrain-epoch-05.ckpt
python b_instruct_tuning.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1 --seed 1 --pretrain_ckpt checkpoints/leaf-pretrain-epoch-10.ckpt

python3 c_inference.py --config configs/LEAF_mpnet.yaml --gpu 0 --bs 512 --level 0,1,2 --ckpt 100
