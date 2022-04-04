#CUDA_VISIBLE_DEVICES=-1 python train_curriculum.py --config configs/maml/mini_grid_curriculum.yaml --output-folder maml-mini-grid --seed 1 --num-workers 8 --wandb --name="mini-grid"
python train_curriculum.py --config configs/maml/mini_grid_curriculum.yaml --output-folder maml-mini-grid --seed 1 --num-workers 8 --wandb --name="mini-grid-trpo"
