from easydict import EasyDict


# (1) refer to /home/gkf/project/ALPRO/src/configs/config.py & /home/gkf/project/ALPRO/config_release/msrvtt_ret.json
# (2) we changed the dir `config_release/` to `Alpro_config_release/`
video_retrieval_configs = {'debug': False, 'data_ratio': 1.0, 'model_config': 'Alpro_config_release/base_model.json', 'tokenizer_dir': 'ext/bert-base-uncased/', 'output_dir': None, 'max_txt_len': 40, 'img_pixel_mean': [0.48145466, 0.4578275, 0.40821073], 'img_pixel_std': [0.26862954, 0.26130258, 0.27577711], 'img_input_format': 'RGB', 'max_n_example_per_group': 1, 'fps': 1, 'num_frm': 8, 'frm_sampling_strategy': 'rand', 'train_n_clips': 1, 'score_agg_func': 'mean', 'random_sample_clips': 1, 'train_batch_size': 8, 'val_batch_size': 8, 'gradient_accumulation_steps': 1, 'learning_rate': 2.5e-05, 'log_interval': 500, 'num_valid': 20, 'min_valid_steps': 100, 'save_steps_ratio': 0.01, 'num_train_epochs': 5, 'optim': 'adamw', 'betas': [0.9, 0.98], 'decay': 'linear', 'dropout': 0.1, 'weight_decay': 0.001, 'grad_norm': 5.0, 'warmup_ratio': 0.1, 'transformer_lr_mul': 1.0, 'step_decay_epochs': None, 'model_type': 'pretrain', 'timesformer_model_cfg': '', 'clip_init': 0, 'bert_weights_path': None, 'inference_model_step': -1, 'do_inference': False, 'inference_split': 'val', 'inference_txt_db': None, 'inference_img_db': None, 'inference_batch_size': 64, 'inference_n_clips': 1, 'seed': 42, 'fp16': False, 'n_workers': 4, 'pin_mem': True, 'config': 'Alpro_config_release/msrvtt_ret.json', 'eval_retrieval_batch_size': 256, 'train_datasets': [{'name': 'msrvtt', 'txt': 'data/msrvtt_ret/txt/train.jsonl', 'img': 'data/msrvtt_ret/videos'}], 'val_datasets': [{'name': 'msrvtt_retrieval', 'txt': 'data/msrvtt_ret/txt/val.jsonl', 'img': 'data/msrvtt_ret/videos'}], 'crop_img_size': 224, 'resize_size': 256, 'visual_model_cfg': 'Alpro_config_release/timesformer_divst_8x32_224_k600.json', 'num_workers': 4} 

video_retrieval_configs = EasyDict(video_retrieval_configs) 

video_retrieval_configs.update({
    'e2e_weights_path': 'Alpro_weights/alpro_pretrained_ckpt.pt'
})
