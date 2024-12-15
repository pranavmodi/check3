nohup python donut_adapt.py --batch_size 2 --load_saved > output.log 2>&1 &

tensorboard --logdir your_checkpoint_dir --export_data ./eval_logs


python donut_adapt.py --checkpoint_path /path/to/checkpoint