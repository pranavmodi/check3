nohup python donut_adapt.py --push_to_hub --hub_model_id="PranavMoD/donut-base-sroie" --batch_size 2 --load_saved --checkpoint_path donut-base-sroie/checkpoint-40 > output.log 2>&1 &

tensorboard --logdir donut-base-sroie --export_data ./eval_logs


python donut_adapt.py 


python donut_adapt.py --push_to_hub --hub_model_id="PranavMoD/donut-base-sroie" --batch_size 2