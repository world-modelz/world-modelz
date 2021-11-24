# Experiment with 2 downscaling stpes

1. trained AE for 42500 steps
2. trained SOM for 8000 steps
3. fintuned AE+SOM for 40000 steps


### Commands:
```
python train_ae.py --downscale_steps 2 --embedding_dim 64 --lr 0.0002 --manual_seed 42 --tags downscale_2 --loss_fn SmoothL1 --device-index 1 --name auto_encoder_ds2 --wand

python train_som.py --downscale_steps 2 --embedding_dim 64 --tags downscale_2 --device-index 1 --ae_checkpoint auto_encoder_ds2_checkpoint_0042500.pth --eta_begin 0.3 --eta_end 0.1 --sigma_begin 64 --sigma_end 0.25  --name som_ds2 --wandb

python finetune_ae.py --som_checkpoint som_ds2_som_checkpoint_0008000.pth --latent_loss_weight 0.1 --lr 0.0002 --som_adapt_batch 8 --max_epochs 5 --loss_fn SmoothL1  --device-index 0 --batch_size 32 --name som_ds2_8k_1 --wandb

```

### Generate diffusion input:
```
python create_diffusion_dataset.py --device-index 1 --checkpoint experiments/ds2/som_ds2_8k_1_checkpoint_0040000.pth --max_examples 1000 --batch_size 64 --dataset_fn diffusion_input_1k.pth
python create_diffusion_dataset.py --device-index 1 --checkpoint experiments/ds2/som_ds2_8k_1_checkpoint_0040000.pth --max_examples 10000 --batch_size 64 --dataset_fn diffusion_input_10k.pth
python create_diffusion_dataset.py --device-index 1 --checkpoint experiments/ds2/som_ds2_8k_1_checkpoint_0040000.pth --max_examples 100000 --batch_size 64 --dataset_fn diffusion_input_100k.pth
python create_diffusion_dataset.py --device-index 1 --checkpoint experiments/ds2/som_ds2_8k_1_checkpoint_0040000.pth --batch_size 64 --dataset_fn diffusion_input_all.pth
```