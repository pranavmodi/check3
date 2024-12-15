nohup python donut_adapt.py --batch_size 2 --load_saved --checkpoint_path donut-base-sroie/checkpoint-40 > output.log 2>&1 &

tensorboard --logdir your_checkpoint_dir --export_data ./eval_logs
