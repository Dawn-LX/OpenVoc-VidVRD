
import root_path

import random
import numpy as np
import argparse
import os 
import json
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch



from models.TrajClsModel import OpenVocTrajCls

from models.RelationClsModel_v3 import AlproVisual_with_FixedPrompt,VidVRDII_FixedPrompt


from dataloaders.dataset_vidvrd_v2 import VidVRDUnifiedDataset,VidVRDUnifiedDataset_GIoU
from utils.config_parser import parse_config_py
from utils.utils_func import get_to_device_func
from utils.logger import LOGGER,add_log_to_file
from utils.evaluate import EvalFmtCvtor
from VidVRDhelperEvalAPIs import eval_visual_relation,evaluate_v2

def load_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x


def modify_state_dict(state_dict):
    # NOTE This function is temporary
    
    text_embeddings = state_dict.pop("text_embeddings") # (36,dim_emb)
    background_emb = state_dict.pop("background_emb")  # (dim_emb,)
    class_embeddings = torch.cat((background_emb[None,:],text_embeddings[1:,:]),dim=0)

    state_dict.update({"class_embeddings":class_embeddings})

    return state_dict


def eval_relation(
    model_class,
    dataset_class,
    args
):
    
    output_dir = args.output_dir
    save_tag = args.save_tag
    if args.output_dir is None:
        output_dir = os.path.dirname(args.cfg_path)
    log_dir = os.path.join(args.output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    add_log_to_file(log_path)
    LOGGER.info("use args:{}".format(args))


    configs = parse_config_py(args.cfg_path)
    if args.eval_type == "SGDet":
        dataset_cfg = configs["eval_dataset_cfg"]
    elif args.eval_type == "PredCls" or args.eval_type == "SGCls":
        dataset_cfg = configs["GTeval_dataset_cfg"]
    
    configs["association_cfg"]["association_n_workers"] = args.asso_n_workers
    model_traj_cfg = configs["model_traj_cfg"]
    model_pred_cfg = configs["model_pred_cfg"]
    eval_cfg = configs["eval_cfg"]
    pred_topk = eval_cfg["pred_topk"]
    device = torch.device("cuda")
        

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model_traj config: {}".format(model_traj_cfg))
    LOGGER.info("model_pred config: {}".format(model_pred_cfg))
    LOGGER.info("evaluate config: {}".format(eval_cfg))

    


    model_traj = OpenVocTrajCls(model_traj_cfg,is_train=False)
    LOGGER.info(f"loading check point from {args.ckpt_path_traj}")
    check_point = torch.load(args.ckpt_path_traj,map_location=torch.device('cpu'))
    state_dict = check_point["model_state_dict"]
    model_traj = model_traj.to(device)
    state_dict = modify_state_dict(state_dict)
    model_traj.load_state_dict(state_dict)
    model_traj.eval()
    model_traj.reset_classifier_weights(args.classifier_split_traj)

    model_pred = model_class(model_pred_cfg)
    LOGGER.info(f"loading check point from {args.ckpt_path_pred}")
    check_point = torch.load(args.ckpt_path_pred,map_location=torch.device("cpu"))
    state_dict = check_point["model_state_dict"]
    model_pred = model_pred.to(device)
    model_pred.load_state_dict(state_dict)
    model_pred.eval()
    if hasattr(model_pred,"reset_classifier_weights"):
        model_pred.reset_classifier_weights(args.classifier_split_pred)
        # model_pred.reset_classifier_weights("all")  # args.classifier_split_pred


    LOGGER.info("preparing dataloader...")
    dataset = dataset_class(**dataset_cfg)
    to_device_func = get_to_device_func(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x : x[0] ,
        num_workers = 4,
        drop_last= False,
        shuffle= False,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "batch_size==1, len(dataset)=={} == len(dataloader)=={}".format(
            dataset_len,dataloader_len
        )
    )

    LOGGER.info("start evaluating:")
    LOGGER.info("use config: {}".format(args.cfg_path))
    LOGGER.info("eval_type: {}".format(args.eval_type))
    
    score_merge = "mean"
    convertor = EvalFmtCvtor(
        "vidvrd",
        args.enti_cls_split_info_path,
        args.pred_cls_split_info_path,
        score_merge=score_merge,
        segment_cvt=True
    )    
    infer_results_for_save = dict()
    for data in tqdm(dataloader):
        # for simplicity, we set batch_size = 1 at inference time
        # seg_tag,det_traj_info,traj_pair_info,rel_pos_feat,labels

        (
            seg_tag,
            det_traj_info,
            traj_embds,
            traj_ids_aft_filter,
            rel_pos_feat,
            labels
        ) = data

        det_feats  = det_traj_info["features"]
        traj_bboxes = det_traj_info["bboxes"]
        traj_starts = det_traj_info["fstarts"]
        n_det = det_feats.shape[0]

        input_data = (
            det_feats,
            traj_embds,
            rel_pos_feat
        )
        input_data = tuple(to_device_func(x) for x in input_data)
        
        with torch.no_grad():
            if args.eval_type == "SGDet" or args.eval_type == "SGCls":
                traj_scores,traj_cls_ids = model_traj.forward_inference_bsz1(input_data[0])  # (n_det,) # before filter
            elif  args.eval_type == "PredCls":
                traj_scores = torch.ones(size=(n_det,),device=device)
                traj_cls_ids = det_traj_info["cls_ids"].to(device)
            # traj_scores,traj_cls_ids = model_traj.forward_inference_bsz1(input_data[0])  # (n_det,) # before filter
            p_scores,p_clsids,pair_ids = model_pred.forward_inference_bsz1(input_data,args.classifier_split_pred,pred_topk) # (n_pair,k)

        s_ids = pair_ids[:,0]  # (n_pair,)
        o_ids = pair_ids[:,1] 

        s_clsids = traj_cls_ids[s_ids]  # (n_pair,)
        o_clsids = traj_cls_ids[o_ids]
        s_scores = traj_scores[s_ids]
        o_scores = traj_scores[o_ids]

        n_pair,k = p_clsids.shape
        triplet_scores = torch.stack([
            p_scores.reshape(-1),
            s_scores[:,None].repeat(1,k).reshape(-1),
            o_scores[:,None].repeat(1,k).reshape(-1)
        ],dim=-1) # (n_pair*k,3)
        triplet_5tuple = torch.stack([
            p_clsids.reshape(-1),
            s_clsids[:,None].repeat(1,k).reshape(-1),
            o_clsids[:,None].repeat(1,k).reshape(-1),
            s_ids[:,None].repeat(1,k).reshape(-1),
            o_ids[:,None].repeat(1,k).reshape(-1),
        ],dim=-1) #  shape == (n_pair*k,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]

        # pred_probs_all = pred_probs_all.cpu()  # # (n_pair,132)
        # pred_probs_all = pred_probs_all[:,None,:].repeat(1,k,1).reshape(n_pair*k,-1)  # (n_pair,k,132)  --> (n_pair*k, 132)
        # assert pred_probs_all.shape == (n_pair*k,132)

        infer_results_for_save[seg_tag] = {
            "traj_bboxes":[tb.cpu().clone() for tb in traj_bboxes],
            "traj_starts": traj_starts.cpu().clone(),
            "triplet_scores":triplet_scores.cpu().clone(),
            "triplet_5tuple":triplet_5tuple.cpu().clone(),
            # "pred_probs_all":pred_probs_all.cpu().clone()
        }

    LOGGER.info("start to convert infer_results to json_format for eval ... score_merge=\'{}\'".format(score_merge))
    relation_results = dict()
    str_to_write = defaultdict(list)
    for seg_tag,results in tqdm(infer_results_for_save.items()):

        traj_bboxes = results["traj_bboxes"]
        traj_starts = results["traj_starts"]
        triplet_scores = results["triplet_scores"]
        triplet_5tuple = results["triplet_5tuple"]
        # pred_probs_all = results["pred_probs_all"]

        result_per_seg = convertor.to_eval_json_format(
            seg_tag,
            triplet_5tuple,
            triplet_scores,
            traj_bboxes,
            traj_starts,
            triplets_topk=eval_cfg["return_triplets_topk"],
            # debug_infos = {"pred_probs_all":pred_probs_all}
        )
        relation_results.update(result_per_seg)

        if args.segment_eval:  #### for debug
            merged_scores = convertor.score_merge(triplet_scores,dim=-1)  # (n_pair,)
            ## select top50 for string in txt
            topkids = merged_scores.argsort(descending=True)[:50]  # (k,), e.g., k=200
            merged_scores = merged_scores[topkids].tolist()
            triplet_scores = triplet_scores[topkids,:].tolist()
            triplet_5tuple = triplet_5tuple[topkids,:].tolist()

            for score,scores,tuple5 in zip(merged_scores,triplet_scores,triplet_5tuple):
                p_s,s_s,o_s = scores  
                p,s,o,sid,oid = tuple5
                tuple5 = [convertor.enti_id2cls[s],convertor.pred_id2cls[p],convertor.enti_id2cls[o],sid,oid]
                spo_scores = "({:.4f},{:.4f},{:.4f})".format(s_s,p_s,o_s)
                str_ = "score:{:.4f} spo_scores:{} 5tuple:{}".format(score,spo_scores,tuple5)
                str_to_write[seg_tag].append(str_)

    if not args.segment_eval:
        LOGGER.info("start relation association ..., using config : {}".format(configs["association_cfg"]))
        relation_results = relation_association(configs["association_cfg"],relation_results)

    hit_infos = _eval_relation_detection_openvoc(
        args,
        prediction_results=relation_results,
        rt_hit_infos=True
    )

    if args.segment_eval:  ####### for debug
        str_to_write_ = []
        for seg_tag, strings in str_to_write.items():
            try:
                hit_scores,gt2hit_ids = hit_infos[seg_tag]
            except KeyError:
                for idx,str_ in enumerate(strings):
                    str_ = "det_id={} ".format(idx) + str_
                    str_ += "no gt"
                    str_to_write_.append(
                        seg_tag + " " + str_ + "\n"
                    )
                continue

            # top50 of hit_scores has the same order as strings
            for idx,str_ in enumerate(strings):
                str_ = "det_id={} ".format(idx) + str_
                if hit_scores[idx] > 0:
                    str_ =  str_  + "is_hit: {:.4f}".format(hit_scores[idx])
                str_to_write_.append(
                    seg_tag + " " + str_ + "\n"
                )
        

        save_path = os.path.join(output_dir,"relation_results_{}.txt".format(save_tag))
        LOGGER.info("save txt into {}".format(save_path))
        with open(save_path,"w") as f:
            f.writelines(str_to_write_)

    save_path = os.path.join(output_dir,f"VidVRDtest_hit_infos_{save_tag}.pkl")
    LOGGER.info("save hit_infos to {}".format(save_path))
    with open(save_path,'wb') as f:
        pickle.dump(hit_infos,f)
    LOGGER.info("hit_infos saved.")
    
    if args.save_infer_results:
        save_path = os.path.join(output_dir,"infer_results_{}.pkl".format(save_tag))
        LOGGER.info("save infer_results to {}".format(save_path))
        with open(save_path,"wb") as f:
            pickle.dump(infer_results_for_save,f)
        LOGGER.info("results saved.")

    if args.save_json_results:
        save_path = os.path.join(output_dir,f"VidVRDtest_relation_results_{save_tag}.json")
        LOGGER.info("save results to {}".format(save_path))
        LOGGER.info("saving ...")
        with open(save_path,'w') as f:
            json.dump(relation_results,f)
        LOGGER.info("results saved.")
    
    LOGGER.info(f"log saved at {log_path}")
    LOGGER.handlers.clear()





def eval_relation_for_AlproVisual_wo_train(
    model_class,
    dataset_class,
    args
):
    output_dir = args.output_dir
    save_tag = args.save_tag
    if args.output_dir is None:
        output_dir = os.path.dirname(args.cfg_path)
    log_dir = os.path.join(args.output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    add_log_to_file(log_path)
    LOGGER.info("use args:{}".format(args))


    configs = parse_config_py(args.cfg_path)
    if args.eval_type == "SGDet":
        dataset_cfg = configs["eval_dataset_cfg"]
    elif args.eval_type == "PredCls" or args.eval_type == "SGCls":
        dataset_cfg = configs["GTeval_dataset_cfg"]
    
    configs["association_cfg"]["association_n_workers"] = args.asso_n_workers
    model_traj_cfg = configs["model_traj_cfg"]
    model_pred_cfg = configs["model_pred_cfg"]
    eval_cfg = configs["eval_cfg"]
    pred_topk = eval_cfg["pred_topk"]
    device = torch.device("cuda")
        

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model_traj config: {}".format(model_traj_cfg))
    LOGGER.info("model_pred config: {}".format(model_pred_cfg))
    LOGGER.info("evaluate config: {}".format(eval_cfg))


    model_traj = OpenVocTrajCls(model_traj_cfg,is_train=False)
    LOGGER.info(f"loading check point from {args.ckpt_path_traj}")
    check_point = torch.load(args.ckpt_path_traj,map_location=torch.device('cpu'))
    state_dict = check_point["model_state_dict"]
    model_traj = model_traj.to(device)
    state_dict = modify_state_dict(state_dict)
    model_traj.load_state_dict(state_dict)
    model_traj.eval()
    model_traj.reset_classifier_weights(args.classifier_split_traj)

    model_pred = model_class(model_pred_cfg)
    assert isinstance(model_pred,AlproVisual_with_FixedPrompt)
    model_pred = model_pred.to(device)
    model_pred.eval()
    

    LOGGER.info("preparing dataloader...")
    dataset = dataset_class(**dataset_cfg)
    to_device_func = get_to_device_func(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x : x[0] ,
        num_workers = 4,
        drop_last= False,
        shuffle= False,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "batch_size==1, len(dataset)=={} == len(dataloader)=={}".format(
            dataset_len,dataloader_len
        )
    )

    LOGGER.info("start evaluating:")
    LOGGER.info("use config: {}".format(args.cfg_path))
    LOGGER.info("eval_type: {}".format(args.eval_type))
    
    score_merge = "mean"
    convertor = EvalFmtCvtor(
        "vidvrd",
        args.enti_cls_split_info_path,
        args.pred_cls_split_info_path,
        score_merge=score_merge,
        segment_cvt=True
    )    
    infer_results_for_save = dict()
    for data in tqdm(dataloader):
        # for simplicity, we set batch_size = 1 at inference time
        # seg_tag,det_traj_info,traj_pair_info,rel_pos_feat,labels

        (
            seg_tag,
            det_traj_info,
            traj_embds,
            traj_ids_aft_filter,
            rel_pos_feat,
            labels
        ) = data

        det_feats  = det_traj_info["features"]
        traj_bboxes = det_traj_info["bboxes"]
        traj_starts = det_traj_info["fstarts"]
        n_det = det_feats.shape[0]

        input_data = (
            det_feats,
            traj_embds,
            rel_pos_feat
        )
        input_data = tuple(to_device_func(x) for x in input_data)
        
        with torch.no_grad():
            if args.eval_type == "SGDet" or args.eval_type == "SGCls":
                traj_scores,traj_cls_ids = model_traj.forward_inference_bsz1(input_data[0])  # (n_det,) # before filter
            elif  args.eval_type == "PredCls":
                traj_scores = torch.ones(size=(n_det,),device=device)
                traj_cls_ids = det_traj_info["cls_ids"].to(device)
            # traj_scores,traj_cls_ids = model_traj.forward_inference_bsz1(input_data[0])  # (n_det,) # before filter
            p_scores,p_clsids,pair_ids = model_pred.forward_inference_bsz1(input_data,args.classifier_split_pred,pred_topk) # (n_pair,k)

        s_ids = pair_ids[:,0]  # (n_pair,)
        o_ids = pair_ids[:,1] 

        s_clsids = traj_cls_ids[s_ids]  # (n_pair,)
        o_clsids = traj_cls_ids[o_ids]
        s_scores = traj_scores[s_ids]
        o_scores = traj_scores[o_ids]

        n_pair,k = p_clsids.shape
        triplet_scores = torch.stack([
            p_scores.reshape(-1),
            s_scores[:,None].repeat(1,k).reshape(-1),
            o_scores[:,None].repeat(1,k).reshape(-1)
        ],dim=-1) # (n_pair*k,3)
        triplet_5tuple = torch.stack([
            p_clsids.reshape(-1),
            s_clsids[:,None].repeat(1,k).reshape(-1),
            o_clsids[:,None].repeat(1,k).reshape(-1),
            s_ids[:,None].repeat(1,k).reshape(-1),
            o_ids[:,None].repeat(1,k).reshape(-1),
        ],dim=-1) #  shape == (n_pair*k,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]

        # pred_probs_all = pred_probs_all.cpu()  # # (n_pair,132)
        # pred_probs_all = pred_probs_all[:,None,:].repeat(1,k,1).reshape(n_pair*k,-1)  # (n_pair,k,132)  --> (n_pair*k, 132)
        # assert pred_probs_all.shape == (n_pair*k,132)

        infer_results_for_save[seg_tag] = {
            "traj_bboxes":[tb.cpu().clone() for tb in traj_bboxes],
            "traj_starts": traj_starts.cpu().clone(),
            "triplet_scores":triplet_scores.cpu().clone(),
            "triplet_5tuple":triplet_5tuple.cpu().clone(),
            # "pred_probs_all":pred_probs_all.cpu().clone()
        }

    LOGGER.info("start to convert infer_results to json_format for eval ... score_merge=\'{}\'".format(score_merge))
    relation_results = dict()
    str_to_write = defaultdict(list)
    for seg_tag,results in tqdm(infer_results_for_save.items()):

        traj_bboxes = results["traj_bboxes"]
        traj_starts = results["traj_starts"]
        triplet_scores = results["triplet_scores"]
        triplet_5tuple = results["triplet_5tuple"]
        # pred_probs_all = results["pred_probs_all"]

        result_per_seg = convertor.to_eval_json_format(
            seg_tag,
            triplet_5tuple,
            triplet_scores,
            traj_bboxes,
            traj_starts,
            triplets_topk=eval_cfg["return_triplets_topk"],
            # debug_infos = {"pred_probs_all":pred_probs_all}
        )
        relation_results.update(result_per_seg)

        if args.segment_eval:  #### for debug
            merged_scores = convertor.score_merge(triplet_scores,dim=-1)  # (n_pair,)
            ## select top50 for string in txt
            topkids = merged_scores.argsort(descending=True)[:50]  # (k,), e.g., k=200
            merged_scores = merged_scores[topkids].tolist()
            triplet_scores = triplet_scores[topkids,:].tolist()
            triplet_5tuple = triplet_5tuple[topkids,:].tolist()

            for score,scores,tuple5 in zip(merged_scores,triplet_scores,triplet_5tuple):
                p_s,s_s,o_s = scores  
                p,s,o,sid,oid = tuple5
                tuple5 = [convertor.enti_id2cls[s],convertor.pred_id2cls[p],convertor.enti_id2cls[o],sid,oid]
                spo_scores = "({:.4f},{:.4f},{:.4f})".format(s_s,p_s,o_s)
                str_ = "score:{:.4f} spo_scores:{} 5tuple:{}".format(score,spo_scores,tuple5)
                str_to_write[seg_tag].append(str_)

    if not args.segment_eval:
        LOGGER.info("start relation association ..., using config : {}".format(configs["association_cfg"]))
        relation_results = relation_association(configs["association_cfg"],relation_results)

    hit_infos = _eval_relation_detection_openvoc(
        args,
        prediction_results=relation_results,
        rt_hit_infos=True
    )

    if args.segment_eval:  ####### for debug
        str_to_write_ = []
        for seg_tag, strings in str_to_write.items():
            try:
                hit_scores,gt2hit_ids = hit_infos[seg_tag]
            except KeyError:
                for idx,str_ in enumerate(strings):
                    str_ = "det_id={} ".format(idx) + str_
                    str_ += "no gt"
                    str_to_write_.append(
                        seg_tag + " " + str_ + "\n"
                    )
                continue

            # top50 of hit_scores has the same order as strings
            for idx,str_ in enumerate(strings):
                str_ = "det_id={} ".format(idx) + str_
                if hit_scores[idx] > 0:
                    str_ =  str_  + "is_hit: {:.4f}".format(hit_scores[idx])
                str_to_write_.append(
                    seg_tag + " " + str_ + "\n"
                )
        

        save_path = os.path.join(output_dir,"relation_results_{}.txt".format(save_tag))
        LOGGER.info("save txt into {}".format(save_path))
        with open(save_path,"w") as f:
            f.writelines(str_to_write_)

    save_path = os.path.join(output_dir,f"VidVRDtest_hit_infos_{save_tag}.pkl")
    LOGGER.info("save hit_infos to {}".format(save_path))
    with open(save_path,'wb') as f:
        pickle.dump(hit_infos,f)
    LOGGER.info("hit_infos saved.")
    
    if args.save_infer_results:
        save_path = os.path.join(output_dir,"infer_results_{}.pkl".format(save_tag))
        LOGGER.info("save infer_results to {}".format(save_path))
        with open(save_path,"wb") as f:
            pickle.dump(infer_results_for_save,f)
        LOGGER.info("results saved.")

    if args.save_json_results:
        save_path = os.path.join(output_dir,f"VidVRDtest_relation_results_{save_tag}.json")
        LOGGER.info("save results to {}".format(save_path))
        LOGGER.info("saving ...")
        with open(save_path,'w') as f:
            json.dump(relation_results,f)
        LOGGER.info("results saved.")
    
    LOGGER.info(f"log saved at {log_path}")
    LOGGER.handlers.clear()



def relation_association(config,segment_predictions):
    import multiprocessing
    from utils.association import parallel_association,greedy_graph_association,greedy_relation_association,nms_relation_association

    '''
    this func is modified based on the func `detect` in VidVRD-II, refer to `/home/gkf/project/VidVRD-II/main.py`
    segment_predictions: refer to `convertor.to_eval_json_format` for its format
    '''
    
    segment_tags = list(segment_predictions.keys())
    segment_prediction_groups = defaultdict(dict)
    for seg_tag in sorted(segment_tags):
        video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
        fstart,fend = int(fstart),int(fend)
        segment_prediction_groups[video_name][(fstart,fend)] = segment_predictions[seg_tag]
    video_name_list = sorted(list(segment_prediction_groups.keys()))

    # video-level visual relation detection by relational association
    print('start {} relation association using {} workers'.format(config['association_algorithm'], config['association_n_workers']))
    if config['association_algorithm'] == 'greedy':
        algorithm = greedy_relation_association
    elif config['association_algorithm'] == 'nms':
        algorithm = nms_relation_association
    elif config['association_algorithm'] == 'graph':
        algorithm = greedy_graph_association
    else:
        raise ValueError(config['association_algorithm'])

    video_relations = {}
    if config.get('association_n_workers', 0) > 0:
        with tqdm(total=len(video_name_list)) as pbar:
            pool = multiprocessing.Pool(processes=config['association_n_workers'])
            for vid in video_name_list:
                video_relations[vid] = pool.apply_async(parallel_association,
                        args=(vid, algorithm, segment_prediction_groups[vid], config),
                        callback=lambda _: pbar.update())
            pool.close()
            pool.join()
        for vid in video_relations.keys():
            res = video_relations[vid].get()
            video_relations[vid] = res
    else:
        for vid in tqdm(video_name_list):
            res = algorithm(segment_prediction_groups[vid], **config)
            video_relations[vid] = res

    return video_relations


def _eval_relation_detection_openvoc(
    args,
    prediction_results=None,
    rt_hit_infos = False,
):
    '''
    NOTE this func is only support for VidVRD currently
    '''
    if prediction_results is None:
        LOGGER.info("loading json results from {}".format(args.json_results_path))
        prediction_results = load_json(args.json_results_path)
        LOGGER.info("Done.")
    else:
        assert args.json_results_path is None


    LOGGER.info("filter gt triplets with traj split: {}, predicate split: {}".format(args.target_split_traj,args.target_split_pred))
    traj_cls_info = load_json(args.enti_cls_split_info_path)
    pred_cls_info = load_json(args.pred_cls_split_info_path)
    traj_categories = [c for c,s in traj_cls_info["cls2split"].items() if (s == args.target_split_traj) or args.target_split_traj=="all"]
    traj_categories = set([c for c in traj_categories if c != "__background__"])
    pred_categories = [c for c,s in pred_cls_info["cls2split"].items() if (s == args.target_split_pred) or args.target_split_pred=="all"]
    pred_categories = set([c for c in pred_categories if c != "__background__"])

    if args.segment_eval:
        gt_relations = load_json(args.segment_gt_json)
    else:
        gt_relations = load_json(args.gt_json)

    gt_relations_ = defaultdict(list)
    for vsig,relations in gt_relations.items(): # same format as prediction results json, refer to `VidVRDhelperEvalAPIs/README.md`
        for rel in relations:
            s,p,o = rel["triplet"]
            if not ((s in traj_categories) and (p in pred_categories) and (o in traj_categories)):
                continue
            gt_relations_[vsig].append(rel)
    gt_relations = gt_relations_
    if rt_hit_infos:
        mean_ap, rec_at_n, mprec_at_n,hit_infos = evaluate_v2(gt_relations,prediction_results,viou_threshold=0.5)
    else:
        mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,prediction_results,viou_threshold=0.5)
    LOGGER.info(f"mAP:{mean_ap}, Retection Recall:{rec_at_n}, Tagging Precision: {mprec_at_n}")
    LOGGER.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    LOGGER.info('detection recall: {}'.format(rec_at_n))
    LOGGER.info('tagging precision: {}'.format(mprec_at_n))

    if rt_hit_infos:
        return hit_infos



if __name__ == "__main__":
    import sys
    random.seed(111)
    np.random.seed(111)
    torch.random.manual_seed(111)
    
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--ckpt_path_traj", type=str,help="...")
    parser.add_argument("--ckpt_path_pred", type=str,help="...")
    parser.add_argument("--enti_cls_split_info_path", type=str,default="configs/VidVRD_class_spilt_info.json")
    parser.add_argument("--pred_cls_split_info_path", type=str,default="configs/VidVRD_pred_class_spilt_info_v2.json")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--model_class", type=str,help="...")
    parser.add_argument("--dataset_class", type=str,default="VidVRDUnifiedDataset")
    parser.add_argument("--segment_eval", action="store_true",default=False,help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--save_infer_results", action="store_true",default=False,help="...")
    parser.add_argument("--target_split_traj", type=str,default="all",help="...")
    parser.add_argument("--target_split_pred", type=str,default="novel",help="...")    
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--generalized_setting", action="store_true",default=False,help="...")
    parser.add_argument("--ALpro", action="store_true",default=False,help="...")
    parser.add_argument("--eval_type", type=str)
    parser.add_argument("--asso_n_workers", type=int,default=8)

    parser.add_argument("--gt_json", type=str,default="datasets/gt_jsons/VidVRDtest_gts.json",help="...")    
    parser.add_argument("--segment_gt_json", type=str,default="datasets/gt_jsons/VidVRDtest_segment_gts.json",help="...")
    
     
    parser.add_argument("--json_results_path", type=str,help="...")
    args = parser.parse_args()
    
    # assert False
    # TODO add generalized_setting
    if args.generalized_setting:
        # generalized setting is not used in our paper
        args.classifier_split_traj = "all"
        args.classifier_split_traj = "all"
    else:
        args.classifier_split_traj = args.target_split_traj
        args.classifier_split_pred = args.target_split_pred


    dataset_class = eval(args.dataset_class)
    if args.model_class is not None:
        model_class = eval(args.model_class)
    
    
    if args.json_results_path is not None:
        # _eval_relation_detection_openvoc(args)
        # eval_from_file_for_each_predicate(args)
        eval_from_file_for_predicate_groups(args)
        sys.exit(0)
    
    if args.eval_type is None:
        args.save_tag = args.save_tag + "-".join(["PredCls","SGCls","SGDet"])
        if args.segment_eval:
            args.save_tag = args.save_tag + "_SegEval"
        
        for eval_type in ["PredCls","SGCls","SGDet"]:
            args.eval_type = eval_type

            if args.ALpro:
                eval_relation_for_AlproVisual_wo_train(AlproVisual_with_FixedPrompt,dataset_class,args)
            else:
                eval_relation(model_class,dataset_class,args)
    else:
        assert args.eval_type in ["PredCls","SGCls","SGDet"]
        args.save_tag = args.save_tag + "-" + args.eval_type
        if args.segment_eval:
            args.save_tag = args.save_tag + "_SegEval"
        
        if args.ALpro:
                eval_relation_for_AlproVisual_wo_train(AlproVisual_with_FixedPrompt,dataset_class,args)
        else:
            eval_relation(model_class,dataset_class,args)

    

    '''
    export 
    #################### Table-3 ########################

    ### Table-3 (ALPro) AlproVisual_with_FixedPrompt
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python tools/eval_relation_cls.py \
        --ALpro \
        --pred_cls_split_info_path configs/VidVRD_pred_class_spilt_info_v2.json \
        --model_class AlproVisual_with_FixedPrompt  \
        --dataset_class VidVRDUnifiedDataset \
        --cfg_path experiments/RelationCls_VidVRD/vanilla_ALPro_inference_only/cfg_fixed_prompt.py \
        --ckpt_path_traj experiments/old_TrajCls_weights/model_OpenVoc_w15BS128_epoch_50.pth \
        --output_dir experiments/RelationCls_VidVRD/vanilla_ALPro_inference_only \
        --target_split_traj all \
        --target_split_pred all \
        --save_tag TaPa
    
    ### Table-3 (VidVRDII) VidVRDII_FixedPrompt
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python tools/eval_relation_cls.py \
        --pred_cls_split_info_path configs/VidVRD_pred_class_spilt_info_v2.json \
        --model_class VidVRDII_FixedPrompt  \
        --dataset_class VidVRDUnifiedDataset \
        --cfg_path  experiments/RelationCls_VidVRD/VidVRD_II/cfg_fixedSingle.py \
        --ckpt_path_traj experiments/old_TrajCls_weights/model_OpenVoc_w15BS128_epoch_50.pth \
        --ckpt_path_pred  experiments/RelationCls_VidVRD/VidVRD_II/model_bsz32_best_mAP.pth \
        --output_dir  experiments/RelationCls_VidVRD/VidVRD_II \
        --target_split_traj all \
        --target_split_pred novel \
        --save_tag TaPn
    
    
    '''
    