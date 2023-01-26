import torch
import pickle
from .utils_func import load_json, traj_cutoff,dura_intersection_ts

class EvalFmtCvtor(object):
    def __init__(self,
        dataset_type,
        enti_split_info_path,
        pred_split_info_path,
        score_merge="mul",
        segment_cvt=False,
        is_debug=False
    ):
        self.segment_cvt = segment_cvt  
        self.dataset_type = dataset_type.lower()
        
        if score_merge == "mul":
            self.score_merge = torch.prod
        elif score_merge == "mean":
            self.score_merge = torch.mean
        else:
            assert False
        
        self.enti_id2cls = load_json(enti_split_info_path)["id2cls"]  # the key in json is str, e.g, "0":__background__
        self.pred_id2cls = load_json(pred_split_info_path)["id2cls"]
        self.enti_id2cls = {int(k):v for k,v in self.enti_id2cls.items()}
        self.pred_id2cls = {int(k):v for k,v in self.pred_id2cls.items()}

        if is_debug:
            pass 
            # TODO add code for debug
    
    def _reset_vsig(self,vsig):
        if self.dataset_type == "vidor":
            temp = vsig.split('_')  # e.g., "0001_3598080384" or "0001_3598080384-0015-0045"
            assert len(temp) == 2
            vsig = temp[1]
        else: # for vidvrd, e.g., vsig == "ILSVRC2015_train_00005015" or  "ILSVRC2015_train_00005015-0015-0045"
            pass
        
        return vsig
    
    def to_eval_json_format(self,vsig,pr_5tuple,pso_scores,traj_bboxes,traj_starts,triplets_topk=-1,debug_infos=dict()):
        '''
        refer to `VidVRDhelperEvalAPIs/README.md` for submission json format

        vsig: video signature, which represents video_name for segment_cvt=False and segment_tag for segment_cvt=True
        pr_5tuple,      # shape == (n_pair,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
        pr_scores,       # shape == (n_pair,3)
        traj_bboxes,    # list[tensor] , len == n_det 
        traj_starts,    # (n_det,) w.r.t segment f_start

        FIXME   add segment fstart,  seg_fs
        '''

        vsig = self._reset_vsig(vsig)
        if self.segment_cvt:
            video_name, fstart, fend = vsig.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
            fstart,fend = int(fstart),int(fend)
            traj_starts = traj_starts + fstart

        n_pair,_ = pr_5tuple.shape
        if n_pair == 0:
            print("for {}, n_pair==0".format(vsig))
            return {vsig:[]}

        device = traj_starts.device
        lens = torch.as_tensor([tb.shape[0] for tb in traj_bboxes],device=device) # (n_det,)
        duras = torch.stack([traj_starts,traj_starts+lens-1],dim=-1)  # (n_det,2), closed interval,   w.r.t segment
        duras_inter,mask = dura_intersection_ts(duras,duras,broadcast=True) # (n_det,n_det,2), (n_det,n_det)

        debug_names = list(debug_infos.keys())
        if len(debug_names) > 0:
            for name,info in debug_infos.items():
                assert isinstance(info,torch.Tensor)
                assert info.shape[0] == n_pair  # (n_pair,*,...)
        

        pr_scores = self.score_merge(pso_scores,dim=-1)  # (n_pair,)

        if triplets_topk > 0:
            # keep topk triplets (for saving time when doing the mAP evaluation API)
            topkids = pr_scores.argsort(descending=True)[:triplets_topk]  # (k,), e.g., k=200
            pr_scores = pr_scores[topkids]
            pr_5tuple = pr_5tuple[topkids,:]
            pso_scores = pso_scores[topkids,:]
            n_pair,_ = pr_5tuple.shape
            for name in debug_names:
                debug_infos[name] = debug_infos[name][topkids,...]
            
        else:
            # else, send all predictions into the mAP evaluation API
            pass


        if isinstance(pr_5tuple,torch.Tensor):
            pr_5tuple = pr_5tuple.tolist()
        if isinstance(pr_scores,torch.Tensor):
            pr_scores = pr_scores.tolist()
        
        results_per_video = []
        for p_id in range(n_pair):
            pred_catid,subj_catid,obj_catid,subj_tid,obj_tid = pr_5tuple[p_id]
            if not mask[subj_tid,obj_tid]:
                continue
            ori_sub_traj = traj_bboxes[subj_tid]
            ori_obj_traj = traj_bboxes[obj_tid]

            so_dura = duras_inter[subj_tid,obj_tid,:].tolist()
            subj_dura = duras[subj_tid,:].tolist()
            obj_dura = duras[obj_tid,:].tolist()

            so_dura = (so_dura[0],so_dura[1]+1) # convert to end_fid exclusive format, the same bellow
            subj_dura = (subj_dura[0],subj_dura[1]+1)
            obj_dura = (obj_dura[0],obj_dura[1]+1)

            subject_traj = traj_cutoff(ori_sub_traj,subj_dura,so_dura,vsig)
            object_traj = traj_cutoff(ori_obj_traj,obj_dura,so_dura,vsig)
            assert len(subject_traj) == len(object_traj)
            assert len(subject_traj) == so_dura[1] - so_dura[0]

            
            result_per_triplet = dict()
            result_per_triplet["triplet"] = [self.enti_id2cls[subj_catid],self.pred_id2cls[pred_catid],self.enti_id2cls[obj_catid]]
            result_per_triplet["duration"] = so_dura   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["score"] = float(pr_scores[p_id])
            # print("pr_scores[p_id]",pr_scores[p_id])
            result_per_triplet["sub_traj"] = subject_traj.cpu().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = object_traj.cpu().tolist()
            
            ################## for debug #################
            result_per_triplet["triplet_tid"] = (int(subj_tid),int(pred_catid),int(obj_tid))  # 如果用 [s_id,p_id,p_catid,o_id]的话，那肯定是唯一的
            result_per_triplet["pso_scores"] = pso_scores[p_id,:].tolist()
            for name in debug_names:
                result_per_triplet[name] = debug_infos[name][p_id,...].tolist()
            ################## for debug #################
            
            results_per_video.append(result_per_triplet)
        
        # results_per_video = sorted(results_per_video,key=lambda x: x["score"],reverse=True)  # large --> small
        # results_per_video = results_per_video[:100]
        return {vsig : results_per_video}


    
    def to_eval_json_format_posOnly(self,vsig,pr_5tuple,pr_scores,traj_bboxes,traj_starts,triplets_topk=-1,debug_infos=dict()):
        '''
        refer to `VidVRDhelperEvalAPIs/README.md` for submission json format

        vsig: video signature, which represents video_name for segment_cvt=False and segment_tag for segment_cvt=True
        pr_5tuple,      # shape == (n_pair,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]
        pr_scores,       # shape == (n_pair,3)
        traj_bboxes,    # list[tensor] , len == n_det 
        traj_starts,    # (n_det,) w.r.t segment f_start

        FIXME   add segment fstart,  seg_fs
        '''

        vsig = self._reset_vsig(vsig)
        if self.segment_cvt:
            video_name, fstart, fend = vsig.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
            fstart,fend = int(fstart),int(fend)
            traj_starts = traj_starts + fstart

        n_pair,_ = pr_5tuple.shape
        if n_pair == 0:
            print("for {}, n_pair==0".format(vsig))
            return {vsig:[]}

        device = traj_starts.device
        lens = torch.as_tensor([tb.shape[0] for tb in traj_bboxes],device=device) # (n_det,)
        duras = torch.stack([traj_starts,traj_starts+lens-1],dim=-1)  # (n_det,2), closed interval,   w.r.t segment
        duras_inter,mask = dura_intersection_ts(duras,duras,broadcast=True) # (n_det,n_det,2), (n_det,n_det)

        debug_names = list(debug_infos.keys())
        if len(debug_names) > 0:
            for name,info in debug_infos.items():
                assert isinstance(info,torch.Tensor)
                assert info.shape[0] == n_pair  # (n_pair,*,...)
        

        pr_scores = self.score_merge(pr_scores,dim=-1)  # (n_pair,)

        if triplets_topk > 0:
            # keep topk triplets (for saving time when doing the mAP evaluation API)
            topkids = pr_scores.argsort(descending=True)[:triplets_topk]  # (k,), e.g., k=200
            pr_scores = pr_scores[topkids]
            pr_5tuple = pr_5tuple[topkids,:]
            n_pair,_ = pr_5tuple.shape
            for name in debug_names:
                debug_infos[name] = debug_infos[name][topkids,...]
            
        else:
            # else, send all predictions into the mAP evaluation API
            pass


        if isinstance(pr_5tuple,torch.Tensor):
            pr_5tuple = pr_5tuple.tolist()
        if isinstance(pr_scores,torch.Tensor):
            pr_scores = pr_scores.tolist()
        
        results_per_video = []
        for p_id in range(n_pair):
            pred_catid,subj_catid,obj_catid,subj_tid,obj_tid = pr_5tuple[p_id]
            if not mask[subj_tid,obj_tid]:
                continue
            ori_sub_traj = traj_bboxes[subj_tid]
            ori_obj_traj = traj_bboxes[obj_tid]

            so_dura = duras_inter[subj_tid,obj_tid,:].tolist()
            subj_dura = duras[subj_tid,:].tolist()
            obj_dura = duras[obj_tid,:].tolist()

            so_dura = (so_dura[0],so_dura[1]+1) # convert to end_fid exclusive format, the same bellow
            subj_dura = (subj_dura[0],subj_dura[1]+1)
            obj_dura = (obj_dura[0],obj_dura[1]+1)

            subject_traj = traj_cutoff(ori_sub_traj,subj_dura,so_dura,vsig)
            object_traj = traj_cutoff(ori_obj_traj,obj_dura,so_dura,vsig)
            assert len(subject_traj) == len(object_traj)
            assert len(subject_traj) == so_dura[1] - so_dura[0]

            
            result_per_triplet = dict()
            # result_per_triplet["triplet"] = [self.enti_id2cls[subj_catid],self.pred_id2cls[pred_catid],self.enti_id2cls[obj_catid]]
            result_per_triplet["triplet"] = [self.enti_id2cls[subj_catid],"fg",self.enti_id2cls[obj_catid]]
            result_per_triplet["duration"] = so_dura   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["score"] = float(pr_scores[p_id])
            # print("pr_scores[p_id]",pr_scores[p_id])
            result_per_triplet["sub_traj"] = subject_traj.cpu().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = object_traj.cpu().tolist()
            
            ################## for debug #################
            result_per_triplet["triplet_tid"] = (int(subj_tid),int(pred_catid),int(obj_tid))  # 如果用 [s_id,p_id,p_catid,o_id]的话，那肯定是唯一的
            for name in debug_names:
                result_per_triplet[name] = debug_infos[name][p_id,...].tolist()
            ################## for debug #################
            
            results_per_video.append(result_per_triplet)
        
        # results_per_video = sorted(results_per_video,key=lambda x: x["score"],reverse=True)  # large --> small
        # results_per_video = results_per_video[:100]
        return {vsig : results_per_video}




    def to_eval_json_format_v2(self,vsig,pr_5tuple,pr_scores,traj_bboxes,traj_starts,preserve_debug_info=True,triplets_topk=-1):
        '''
        this func without traj_cutoff
        '''

        vsig = self._reset_vsig(vsig)

        n_pair,_ = pr_5tuple.shape
        if n_pair == 0:
            print("for {}, n_pair==0".format(vsig))
            return {vsig:[]}

        lens = torch.as_tensor([tb.shape[0] for tb in traj_bboxes]) # (n_det,)
        duras = torch.stack([traj_starts,traj_starts+lens-1],dim=-1)  # (n_det,2), closed interval
        duras_inter,mask = dura_intersection_ts(duras,duras,broadcast=True) # (n_det,n_det,2), (n_det,n_det)
        
        pr_scores = self.score_merge(pr_scores,dim=-1)  # (n_pair,)

        if triplets_topk > 0:
            # keep topk triplets (for saving time when doing the mAP evaluation API)
            topkids = pr_scores.argsort(descending=True)[:triplets_topk]  # (k,), e.g., k=200
            pr_scores = pr_scores[topkids]
            pr_5tuple = pr_5tuple[topkids,:]
            n_pair,_ = pr_5tuple.shape
        else:
            # else, send all predictions into the mAP evaluation API
            pass


        # pred_duras_float = debug_info
        if isinstance(pr_5tuple,torch.Tensor):
            pr_5tuple = pr_5tuple.tolist()
        if isinstance(pr_scores,torch.Tensor):
            pr_scores = pr_scores.tolist()
        
        results_per_video = []
        for p_id in range(n_pair):
            pred_catid,subj_catid,obj_catid,subj_tid,obj_tid = pr_5tuple[p_id]
            if not mask[subj_tid,obj_tid]:
                continue
            ori_sub_traj = traj_bboxes[subj_tid]
            ori_obj_traj = traj_bboxes[obj_tid]

            so_dura = duras_inter[subj_tid,obj_tid,:].tolist()
            so_dura = (so_dura[0],so_dura[1]+1) # convert to end_fid exclusive format, the same bellow


            
            result_per_triplet = dict()
            result_per_triplet["triplet"] = [self.enti_id2cls[subj_catid],self.pred_id2cls[pred_catid],self.enti_id2cls[obj_catid]]
            result_per_triplet["duration"] = so_dura   # [strat_fid, end_fid)   starting (inclusive) and ending (exclusive) frame ids
            result_per_triplet["score"] = float(pr_scores[p_id])
            result_per_triplet["sub_traj"] = ori_sub_traj.cpu().tolist()     # len == duration_spo[1] - duration_spo[0]
            result_per_triplet["obj_traj"] = ori_obj_traj.cpu().tolist()
            
            ################## for debug #################
            if preserve_debug_info:
                result_per_triplet["triplet_tid"] = (int(subj_tid),int(pred_catid),int(obj_tid))  # 如果用 [s_id,p_id,p_catid,o_id]的话，那肯定是唯一的
                # result_per_triplet["debug_dura"] = {"s":subject_dura,"o":object_dura,"inter":dura_,"pr_float_dura":pr_float_dura}
            ################## for debug #################
            
            results_per_video.append(result_per_triplet)
        
        # results_per_video = sorted(results_per_video,key=lambda x: x["score"],reverse=True)  # large --> small
        # results_per_video = results_per_video[:100]
        return {vsig : results_per_video}

