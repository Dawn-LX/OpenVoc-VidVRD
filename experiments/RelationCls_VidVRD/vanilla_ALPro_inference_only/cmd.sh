TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python tools/eval_relation_cls.py \
        --pred_cls_split_info_path configs/VidVRD_pred_class_spilt_info_v2.json \
        --model_class AlproVisual_with_FixedPrompt  \
        --dataset_class VidVRDUnifiedDataset \
        --cfg_path experiments_RelationCls/_exp_models_v3_TrajBasePredBase/AlproVisual_with_FixedPrompt/cfg_fixed_separate.py \
        --ckpt_path_traj experiments/ALPro_teacher/model_OpenVoc_w15BS128_epoch_50.pth \
        --output_dir experiments_RelationCls/_exp_models_v3_TrajBasePredBase/AlproVisual_with_FixedPrompt/cfg_fixed_separate \
        --target_split_traj all \
        --target_split_pred all \
        --save_tag TaPa