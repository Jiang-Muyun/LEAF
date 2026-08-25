python init_text_embeddings.py all-mpnet-base-v2
python init_text_embeddings.py qwen3-embedding-4B

python a_pretrain.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1

# instruct tuning using the pretrain checkpoint
python b_instruct_tuning.py --config configs/LEAF_mpnet.yaml    --bs 256 --gpu 0,1 --seed 0 --pretrain_ckpt leaf-v1.0-pretrain.ckpt
python b_instruct_tuning.py --config configs/LEAF_qwen3-4b.yaml --bs 256 --gpu 0,1 --seed 0 --pretrain_ckpt leaf-v1.0-pretrain.ckpt

# directly inference using the instruct checkpoint
python3 c_inference.py --config configs/LEAF_mpnet.yaml    --gpu 0 --bs 512 --level 0,1,2 --ckpt leaf-v1.0-instruct-mpnet-base.ckpt
python3 c_inference.py --config configs/LEAF_qwen3-4b.yaml --gpu 0 --bs 512 --level 0,1,2 --ckpt leaf-v1.0-instruct-qwen3-4b.ckpt
