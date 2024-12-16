nohup python donut_adapt.py --push_to_hub --hub_model_id="PranavMoD/donut-base-sroie" --batch_size 2 --load_saved --checkpoint_path donut-base-sroie/checkpoint-40 > output.log 2>&1 &

tensorboard --logdir donut-base-sroie --export_data ./eval_logs


python donut_adapt.py 


python donut_adapt.py --push_to_hub --hub_model_id="PranavMoD/donut-base-sroie" --batch_size 2



python predict.py --model_path donut-base-sroie --processor_path processed_data/processor --image_path path/to/image.jpg


python hf_push.py --checkpoint_dir "donut-base-sroie" --hub_model_id "PranavMoD/donut-base-sroie" --commit_message "Final trained model"


python donut_predict.py --model_path "PranavMod/donut-base-sroie" --processor_path "processor_download" --image_path test/000.jpg --use_hf