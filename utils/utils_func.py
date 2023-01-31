# import warnings
import json
import numpy as np
import torch
import torch.nn.functional as F
import logging
from torchvision.ops import roi_pool as roi_pool2d

def trajid2pairid(n_det): 
    # mask = torch.ones(size=(n_det,n_det),dtype=torch.bool) 
    # mask[range(n_det),range(n_det)] = 0 
    # # print(mask) 
    # pair_ids = mask.nonzero(as_tuple=False) 
    # # print(pair_ids)
    # return pair_ids 

    ### the above code is equivalent to following
    zz = torch.cartesian_prod(torch.as_tensor(range(n_det)),torch.as_tensor(range(n_det)))
    zz = zz[zz[:,0]!=zz[:,1]]
    return zz 

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    code from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions (logits) for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def _focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    code from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions (prob) for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(p,targets,reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss



def load_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x

def get_to_device_func(device):
    def to_device_func(data):
        d = to_device_func.device
        if isinstance(data,torch.Tensor):
            data = data.to(d)            
        elif isinstance(data,list):
            data = [to_device_func(item) for item in data]
        elif isinstance(data,tuple):
            data = tuple(to_device_func(item) for item in data)
        elif isinstance(data,dict):
            data = {k:to_device_func(v) for k,v in data.items()}
        elif isinstance(data,(str,float,int,np.ndarray)) or (data is None):
            pass
        else:
            print(type(data))
                
        return data
    
    to_device_func.device = device

    return to_device_func

def stack_with_padding(tensor_list,dim,value=0,rt_mask=False):
    """
    Example:
        >>> x = torch.randn(3,4)
        >>> y = torch.randn(2,5)
        >>> z = stack_with_padding([x,y],dim=0) # z.shape == (2,3,5)
        >>> print(z)
        tensor([[[ 0.2654,  0.5374, -0.5466, -0.1828,  0.0000],
                [-0.4146, -0.5796, -0.7139,  0.4708,  0.0000],
                [ 1.4727,  0.5511,  0.3228, -1.3286,  0.0000]],

                [[-2.3506,  0.1536, -1.4882,  0.1360,  0.3050],
                [ 0.3862,  0.2438, -0.7124, -0.8490, -1.9474],
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]])
    """
    shape_list = [t.shape for t in tensor_list]
    n_dim = len(shape_list[0])
    max_sp = []  # --> len == n_dim
    for i in range(n_dim):
        max_sp.append(
            max([sp[i] for sp in shape_list])
        )
    aft_pad_list = []
    mask_list = []
    for tensor in tensor_list:
        sp = tensor.shape
        pad_n = [m-s for m,s in zip(max_sp,sp)]
        pad_n.reverse()
        pad_size = []
        for pn in pad_n:
            pad_size += [0,pn]
        aft_pad_list.append(
            torch.constant_pad_nd(tensor,pad_size,value=value)
        )
        if rt_mask:
            mask = torch.ones(tensor.shape,dtype=torch.bool,device=tensor.device)
            mask_list.append(
                torch.constant_pad_nd(mask,pad_size)
            )
    if rt_mask:
        return torch.stack(aft_pad_list,dim=dim),torch.stack(mask_list,dim=dim)
    else:
        return torch.stack(aft_pad_list,dim=dim)

def stack_with_repeat_2d(tensor_list,dim):
    assert len(tensor_list[0].shape) == 2
    device = tensor_list[0].device
    shape_list = [t.shape for t in tensor_list]
    num_rows = torch.tensor([sp[0] for sp in shape_list])
    num_cols = torch.tensor([sp[1] for sp in shape_list])
    # assert num_rows[0]
    if torch.all(num_rows == num_rows[0]):
        max_L = num_cols.max()
        repeat_dim=1
    elif torch.all(num_cols == num_cols[0]):
        max_L = num_rows.max()
        repeat_dim=0
    else:
        assert False
    
    after_repeat = []
    for tensor in tensor_list:
        L = tensor.shape[repeat_dim]
        n_pad = L - (max_L % L)
        ones = [1]*max_L
        zeros = [0]*n_pad
        total = torch.tensor(ones + zeros,device=device)
        total = total.reshape(-1,L)
        repeats_ = total.sum(dim=0)
        after_repeat.append(
            tensor.repeat_interleave(repeats_,dim=repeat_dim)
        )
    return torch.stack(after_repeat,dim=dim)

def merge_consec_fg(segement_list):
    assert isinstance(segement_list,list)
    bg_ratio_th = 0.5
    num_seg = len(segement_list)
    # each segment is assiged with a number, which records the number of bg (0 for consecutive fg segments)
    after_merged_all_lvls = []
    level_1 = [(x,0) for x in segement_list]
    after_merged_all_lvls.append(level_1)

    while True:
        segs_crt_lvl = after_merged_all_lvls[-1]
        num_seg = len(segs_crt_lvl)

        segs_next_lvl = []
        for idx in range(num_seg-1):
            crt_seg,n_bg1 = segs_crt_lvl[idx]
            next_seg,n_bg2 = segs_crt_lvl[idx+1]
            
            span = next_seg[0] - crt_seg[-1] -1
            new_bgs = span if span > 0 else 0 

            num_bgs = n_bg1 + n_bg2 + new_bgs
            merged_seg = sorted(list(set(crt_seg + next_seg)))
            merged_seg = (merged_seg, num_bgs)
            if num_bgs/(len(merged_seg[0])+num_bgs) < bg_ratio_th:
                segs_next_lvl.append(merged_seg)
        if segs_next_lvl == []:
            break
        else:
            after_merged_all_lvls.append(segs_next_lvl)
    
    all_merged_segs = []
    for segs_per_lvl in after_merged_all_lvls:
        # print(segs_per_lvl)
        segs_per_lvl = [x[0] for x in segs_per_lvl]  # drop the `n_bgs`
        all_merged_segs += segs_per_lvl
    return all_merged_segs


def average_to_fixed_length(visual_input,num_sample_clips):
    # original code from https://github.com/microsoft/2D-TAN/blob/e0e7a83ff991e74e07d67d9bcc1be94b1767e9a9/lib/datasets/__init__.py#L30
    
    # num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS   # NUM_SAMPLE_CLIPS == 256
    num_clips = visual_input.shape[0]   # num_clips 一般 > 256, e.g., 432
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    # 这个相当于在 432 个 clips 中均匀采样256个
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

def create_logger(filename='train.log',filemode='a',fmt='%(asctime)s - %(message)s', level=logging.DEBUG):
    """
    reference:https://www.cnblogs.com/nancyzhu/p/8551506.html
    """
    logging.basicConfig(filename=filename,filemode=filemode,format=fmt, level=level)
    logger = logging.getLogger()
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    logger.addHandler(sh)

    return logger

def traj_align_pool(traj_features,inter_dura,roi_outlen,scale):
    # traj_features.shape == (n_traj, max_frames, dim_feat)
    # inter_dura.shape == (2,n_pos_ac,3) format: [tid,start,end]
    n_trajs,_,_ = traj_features.shape
    _,n_pos_ac,_ = inter_dura.shape

    input = traj_features.permute(0,2,1).float()  # shape == (n_trajs, dim_feat, max_frames)
    input = input[...,None]  # shape == (N,C,L,1) == (batch_size,n_channels,Length,1) == (n_trajs, dim_feat, max_frames, 1)

    ## convert inter_dura to 2D-roi format 
    inter_dura = inter_dura.reshape(2*n_pos_ac,-1)  # shape == (2*n_pos_ac,3)
    # print(inter_dura)
    tid = inter_dura[:,None,0]
    assert tid.max() < n_trajs
    tl = torch.constant_pad_nd(inter_dura[:,None,1],pad=(1,0))  
    br = torch.constant_pad_nd(inter_dura[:,None,2],pad=(1,0))  
    rois = torch.cat([tid,tl,br],dim=-1).float()  # shape == (K,5), K=2*n_pos_ac format: [tid,xmin,ymin,xmax,ymax] = [tid,0,ymin,0,ymax] = [tid,0,start,0,end]
    # print(rois)

    output_size = (roi_outlen,1)
    result = roi_pool2d(input,rois,output_size,spatial_scale=scale)  # shape == (K,dim_feat,output_len,1)
    result = result.squeeze(-1).permute(0,2,1)  # shape == (K, output_len,dim_feat),  K=2*n_pos_ac
    result = result.reshape(2,n_pos_ac,roi_outlen,-1)
    return result

def traj_roi_pool(traj_features,inter_dura,adj_mask,roi_outlen,scale):
    # traj_features.shape == (n_trajs, max_frames, dim_feat)
    # inter_dura.shape == (n_trajs, n_anchors, 2)

    input = traj_features.permute(0,2,1)  # shape == (n_trajs, dim_feat, video_len)
    input = input[...,None]  # shape == (N,C,L,1) == (batch_size,n_channels,Length) == (n_trajs, dim_feat, video_len, 1)

    ## convert inter_dura to 2D-roi format 
    # 1.zero-padding        
    tl = torch.constant_pad_nd(inter_dura[:,:,None,0],pad=(1,0))  
    br = torch.constant_pad_nd(inter_dura[:,:,None,1],pad=(1,0))  
    rois = torch.cat([tl,br],dim=-1)  # shape == (n_trajs, n_anchors, 4) format: [xmin,ymin,xmax,ymax] = [0,ymin,0,ymax] = [0,start,0,end]

    # 2. add id  --> shape == (K,5)  K == adj_mask.sum() <= n_trajs*n_anchors
    n_trajs,n_anchors,_ = rois.shape
    rois_tid = torch.tensor(list(range(n_trajs)),device=rois.device)
    rois_tid = rois_tid[:,None,None].repeat(1,n_anchors,1)  # shape == (n_trajs,n_anchors,1)
    # print(rois_tid,rois_tid.shape)
    rois = torch.cat([rois_tid,rois],dim=-1)  # shape == (n_trajs,n_anchors,5)
    rois = rois[adj_mask].float()  # shape == (K, 5)  K == adj_mask.sum() <= n_trajs*n_anchors

    output_size = (roi_outlen,1)
    result = roi_pool2d(input,rois,output_size,spatial_scale=scale)  # shape == (K,dim_feat,output_len,1)
    result = result.squeeze(-1).permute(0,2,1)  # shape == (K, output_len,dim_feat)
    return result

def interpolation_single(vector_l,vector_r,left,right):
    assert left +1 < right  # otherwise we don't need interpolation
    assert len(vector_l.shape) == 1
    assert vector_l.shape == vector_r.shape
    inter_len = right-left-1

    inter_vector = np.linspace(vector_l,vector_r,num=inter_len+2,axis=0)[1:-1]
    return inter_vector

def fill_zeropadding(vectors):
    mask0 = vectors == 0     # shape == (n_box,1024)
    index0 = np.where(np.all(mask0,axis=-1))[0]
    # 没有两帧连续的0填充
    assert np.all(np.diff(index0) > 1) ,"index0={}".format(index0) 
    index_neighbor = index0 - 1
    index_neighbor[index_neighbor == -1] = 1
    vectors[index0] = vectors[index_neighbor]

def linear_interpolation(vectors,frame_ids):
    # vectors.shape == (n_frames,d)  # d=5 for bbox_with_score and d=1024 for RoIfeature
    assert len(vectors.shape) == 2
    frame_ids = np.array(frame_ids)  # shape == (n_frames,)
    frame_id_diff = np.diff(frame_ids)
    cut_point = np.where(frame_id_diff > 1)[0] + 1

    consec_frames = np.split(frame_ids,cut_point)
    consec_vectors = np.split(vectors,cut_point,axis=0)
    num_consecutive = len(consec_frames)

    result_vectors = []
    for i in range(1,num_consecutive,1):
        left_vector = consec_vectors[i-1][-1]  # shape == (4,)
        right_vector = consec_vectors[i][0]
        fill_zeropadding(left_vector)
        fill_zeropadding(right_vector)
        left = consec_frames[i-1][-1]
        right = consec_frames[i][0]
        inter_vectors = interpolation_single(left_vector,right_vector,left,right)
        result_vectors.append(consec_vectors[i-1])
        result_vectors.append(inter_vectors)

    result_vectors.append(consec_vectors[-1])
    result_vectors = np.concatenate(result_vectors,axis=0)
    return result_vectors

def normalize01(x):

    return (x-x.min())/(x.max() - x.min())

def unique_with_idx(tensor):
    assert len(tensor.shape) == 1  # TODO consider muti-dimension
    unique_,counts = torch.unique(tensor,return_counts=True)
    mask = tensor[None,:] == unique_[:,None]
    index_map = mask.nonzero(as_tuple=True)[1]
    index_map = torch.split(index_map,counts.tolist())  # tuple[tensor] len==len(unique), each shape == (count,)

    return unique_,index_map

def unique_with_idx_nd(tensor):
    """
    NOTE consider dim 0 to unique
    tensor.shape == (N,d1,d2,...dk), usually, N > di (and often N is much larger than di)
    TODO consider uset-defined dim to unique
    """
    
    unique_,counts = torch.unique(tensor,return_counts=True,dim=0)
    # unique_.shape == (N_unique,d1,d2,...dk)
    mask = tensor[None,:,...] == unique_[:,None,...]  # shape == (N_unique,N,d1,d2,...dk)
    mask = mask.reshape(mask.shape[0],mask.shape[1],-1)
    mask = torch.all(mask,dim=-1)  # shape == (N_unique,N)
    index_map = mask.nonzero(as_tuple=True)[1]
    index_map = torch.split(index_map,counts.tolist())  # tuple[tensor] len==len(unique), each shape == (count,)

    return unique_,index_map

def dura_intersection_ts(dura1,dura2,broadcast=True):
    """dura1 & dura2 are both closed interval"""
    assert isinstance(dura1,torch.Tensor) and isinstance(dura2,torch.Tensor)
    n1,n2 = dura1.shape[0],dura2.shape[0]
    mask1 = dura1[:,0] <= dura1[:,1]
    mask2 = dura2[:,0] <= dura2[:,1]
    assert mask1.sum() == n1 , "dura1[~mask1,:]={}".format(dura1[~mask1,:])
    assert mask2.sum() == n2 , "dura2[~mask2,:]={}".format(dura2[~mask2,:])
    
    if broadcast:
        inter_s = torch.max(dura1[:,None,0],dura2[None,:,0])
        inter_e = torch.min(dura1[:,None,1],dura2[None,:,1])
        intersection = torch.stack([inter_s,inter_e],dim=-1)
        # print(inter_s,inter_s.shape)
        # print(inter_e,inter_e.shape)
        # print(intersection,intersection.shape)  # shape == (n1,n2,2)
        mask = intersection[:,:,0] <= intersection[:,:,1]   # shape == (n1,n2)
        # print(mask,mask.shape)
        # intersection[~mask] *= -1
        # print(intersection)
    else:
        assert n1 == n2
        inter_s = torch.max(dura1[:,0],dura2[:,0])
        inter_e = torch.min(dura1[:,1],dura2[:,1])
        intersection = torch.stack([inter_s,inter_e],dim=-1) # shape == (n1,2)
        mask = intersection[:,0] <= intersection[:,1]        # shape == (n1,)

    return intersection,mask

def tIoU(duras1,duras2,broadcast=True):
    # duras1.shape == (n1,2)
    # duras2.shape == (n2,2)
    if broadcast:
        mask = (duras1[:,None,1] >= duras2[None,:,0]) * (duras2[None,:,1] >= duras1[:,None,0])  # shape == (n1, n2),dtype=torch.bool
        tiou = (torch.min(duras1[:,None,1],duras2[None,:,1]) - torch.max(duras1[:,None,0],duras2[None,:,0])) \
            / (torch.max(duras1[:,None,1],duras2[None,:,1]) - torch.min(duras1[:,None,0],duras2[None,:,0]))
    else:
        assert duras1.shape == duras2.shape
        mask = (duras1[:,1] >= duras2[:,0]) * (duras2[:,1] >= duras1[:,0])  # shape == (n1,),dtype=torch.bool
        tiou = (torch.min(duras1[:,1],duras2[:,1]) - torch.max(duras1[:,0],duras2[:,0])) \
            / (torch.max(duras1[:,1],duras2[:,1]) - torch.min(duras1[:,0],duras2[:,0]))

    tiou[torch.logical_not(mask)] = 0

    return tiou   # shape == (n1,n2)


def generalized_tIoU(duras1,duras2,broadcast=True):
    # gIoU = IoU - |C\(A U B)| / |C|  \in [-1,1]
    # one-dim IoU (tIoU) is just the above tIoU func without ``tiou[torch.logical_not(mask)] = 0``

    # duras1.shape == (n1,2)
    # duras2.shape == (n2,2)
    if broadcast:
        g_tiou = (torch.min(duras1[:,None,1],duras2[None,:,1]) - torch.max(duras1[:,None,0],duras2[None,:,0])) \
            / (torch.max(duras1[:,None,1],duras2[None,:,1]) - torch.min(duras1[:,None,0],duras2[None,:,0]))
    else:
        assert duras1.shape == duras2.shape
        g_tiou = (torch.min(duras1[:,1],duras2[:,1]) - torch.max(duras1[:,0],duras2[:,0])) \
            / (torch.max(duras1[:,1],duras2[:,1]) - torch.min(duras1[:,0],duras2[:,0]))


    return g_tiou   # shape == (n1,n2)

def vIoU_ts_rel(traj_1,traj_2,dura_1,dura_2):
    '''
    NOTE: this func is deprecated, it is the same as `_vIoU` defined bellow in this .py file
    NOTE: Better to use `vIoU_broadcast` defined bellow, which is a more high-levle wrapper of vIoU
    '''

    """
    dura_1,dura_2 are relative durations, closed interval
    """


    assert isinstance(traj_1,torch.Tensor) and isinstance(traj_2,torch.Tensor)
    assert isinstance(dura_1,torch.Tensor) and isinstance(dura_2,torch.Tensor)
    
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    traj_1 = traj_1[dura_1[0]:dura_1[1]+1,:]
    traj_2 = traj_2[dura_2[0]:dura_2[1]+1,:]
    assert traj_1.shape == traj_2.shape  # shape == (inter_frames, 4)

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)


def vIoU_ts(traj_1,traj_2,dura_1,dura_2):
    '''
    NOTE: this func is deprecated, it is the same as `_vIoU` defined bellow in this .py file
    NOTE: Better to use `vIoU_broadcast` defined bellow, which is a more high-levle wrapper of vIoU
    '''

    """
    dura_1,dura_2 are relative durations, closed interval
    """

    
    # Warning_str = """
    # this `vIoU_ts` has been deprecated. 
    # If you do want to calculate vIoU based on relative duration, please use `vIoU_ts_rel`
    # Otherwise, we suggest using `vIoU_ts_abs` for absolute duration input.
    # """
    # warnings.warn(Warning_str, DeprecationWarning)
    # print(Warning_str)

    ## NOTE 我们没有实现 vIoU_ts_abs ， 如果要实现 vIoU_ts_abs的话，就要在 vIoU_ts_abs 内部执行 dura_intersection_ts 了
    ## 但是我们现在是要在外部执行 dura_intersection_ts， 因为我们要用到 inter_dura, 所以我们对 vIoU_ts 的实现方式采用 relative_dura


    assert isinstance(traj_1,torch.Tensor) and isinstance(traj_2,torch.Tensor)
    assert isinstance(dura_1,torch.Tensor) and isinstance(dura_2,torch.Tensor)
    
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    traj_1 = traj_1[dura_1[0]:dura_1[1]+1,:]
    traj_2 = traj_2[dura_2[0]:dura_2[1]+1,:]
    assert traj_1.shape == traj_2.shape  # shape == (inter_frames, 4)

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)


def bbox_IoU(box1, box2):
    ## https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    # with slight modifications (i.e., add TO_REMOVE = 1 to consider one extra pixel)

    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    box1 = box1.float()
    box2 = box2.float()
    N = box1.size(0)
    M = box2.size(0)
    TO_REMOVE = 1

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb - lt + TO_REMOVE).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+TO_REMOVE) * (box1[:,3]-box1[:,1]+TO_REMOVE)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+TO_REMOVE) * (box2[:,3]-box2[:,1]+TO_REMOVE)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def bbox_GIoU(box1,box2):
    box1 = box1.float()
    box2 = box2.float()
    N = box1.size(0)
    M = box2.size(0)
    TO_REMOVE = 1

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb - lt + TO_REMOVE).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+TO_REMOVE) * (box1[:,3]-box1[:,1]+TO_REMOVE)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+TO_REMOVE) * (box2[:,3]-box2[:,1]+TO_REMOVE)  # [M,]
    
    union = area1[:,None] + area2 - inter  # (N,M)
    iou = inter / union

    u_lt = torch.min(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    u_rb = torch.max(box1[:,None,2:], box2[:,2:])  # [N,M,2]
    u_box = torch.cat([u_lt,u_rb],dim=-1)  # (N,M,4)
    u_box_area = (u_box[:,:,2]-u_box[:,:,0]+TO_REMOVE) * (u_box[:,:,3]-u_box[:,:,1]+TO_REMOVE)  # [N,M]

    giou = iou - (u_box_area - union)/u_box_area

    return giou




def _vIoU(traj_1,traj_2,dura_1,dura_2):
    # NOTE: this func is the same as `vIoU_ts` & `vIoU_ts_rel` defined above in this .py file
    # NOTE: Better to use `vIoU_broadcast` defined bellow, which is a more high-levle wrapper of vIoU

    """
    dura_1,dura_2 are relative durations, closed interval
    """
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    traj_1 = traj_1[dura_1[0]:dura_1[1]+1,:]
    traj_2 = traj_2[dura_2[0]:dura_2[1]+1,:]
    assert traj_1.shape == traj_2.shape  # shape == (inter_frames, 4)

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()  # (inter_frames,) --> scalar
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)



def vIoU_broadcast(trajs_1,trajs_2,starts1,starts2,broadcast=True):
    '''
    trajs_1: list[tensor], len == n1,  each shape == (num_frames, 4)
    trajs_2: list[tensor], len == n2,  each shape == (num_frames, 4)

    starts1  shape == (n1,)
    starts2  shape == (n2,)
    '''

    assert isinstance(starts1,torch.Tensor) and isinstance(starts2,torch.Tensor)
    device = starts1.device
    n1,n2 = len(trajs_1),len(trajs_2)
    lens_1 = torch.as_tensor([traj.shape[0] for traj in trajs_1],device=device) # (n1,)
    lens_2 = torch.as_tensor([traj.shape[0] for traj in trajs_2],device=device) # (n2,)
    duras_1 = torch.stack([starts1,starts1+lens_1-1],dim=-1)  # (n1,2), closed interval
    duras_2 = torch.stack([starts2,starts2+lens_2-1],dim=-1)  # (n2,2)

    duras_inter,mask = dura_intersection_ts(duras_1,duras_2,broadcast=True) # (n1,n2,2), (n1,n2)
    vious = torch.zeros_like(mask,dtype=torch.float)  # (n1,n2)

    if broadcast:
        # duras_inter.shape==(n1,n2,2); mask.shape == (n1,n2)
        rel_duras_1 = duras_inter - duras_1[:,0,None,None]  # (n1,n2,2) # convert to relative duration
        rel_duras_2 = duras_inter - duras_2[None,:,0,None]  # (n1,n2,2)
        pos_ids_1,pos_ids_2 = mask.nonzero(as_tuple=True)  # row, col; positive ids
        for id1,id2 in zip(pos_ids_1.tolist(),pos_ids_2.tolist()):
            dura_1 = rel_duras_1[id1,id2,:]
            dura_2 = rel_duras_2[id1,id2,:]
            traj_1 = trajs_1[id1].float()   # (num_frames,4)
            traj_2 = trajs_2[id2].float()   # (num_frames,4)

            vious[id1,id2] = _vIoU(traj_1,traj_2,dura_1,dura_2)
    else:
        assert n1==n2
        # duras_inter.shape==(n1,2); mask.shape == (n1,)
        rel_duras_1 = duras_inter - duras_1[:,0,None]  # (n1,2) # convert to relative duration
        rel_duras_2 = duras_inter - duras_2[:,0,None]  # (n1,2)
        pos_ids = mask.nonzero(as_tuple=True)[0]  # positive ids
        for idx in pos_ids:
            dura_1 = rel_duras_1[idx,:]
            dura_2 = rel_duras_2[idx,:]
            traj_1 = trajs_1[idx].float()   # (num_frames,4)
            traj_2 = trajs_2[idx].float()   # (num_frames,4)

            vious[idx] = _vIoU(traj_1,traj_2,dura_1,dura_2)


    return vious



def _vPoI(traj,traj_gt,dura,dura_gt):
    # similar pipeline as the _vIoU & vIoU_broadcast calculating
    # vPoI refers to volume Proportion of Intersection, 
    # refer to Video Visual Relation Detection via Iterative Inference ACM MM 2021

    """
    dura_1,dura_2 are relative durations, closed interval
    """
    traj = traj.float()
    traj_gt = traj_gt.float()
    TO_REMOVE = 1
    area = (traj[:, 2] - traj[:, 0] + TO_REMOVE) * (traj[:, 3] - traj[:, 1] + TO_REMOVE)
    # area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)
    
    traj = traj[dura[0]:dura[1]+1,:]
    traj_gt = traj_gt[dura_gt[0]:dura_gt[1]+1,:]
    assert traj.shape == traj_gt.shape  # shape == (inter_frames, 4)

    lt = torch.max(traj[:,:2],traj_gt[:,:2])
    rb = torch.min(traj[:,2:],traj_gt[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / area.sum()



def vPoI_broadcast(trajs,trajs_gt,starts,starts_gt,broadcast=True):
    '''
    similar pipeline as the _vIoU & vIoU_broadcast calculating
    vPoI refers to volume Proportion of Intersection, 
    refer to Video Visual Relation Detection via Iterative Inference ACM MM 2021

    trajs: list[tensor], len == n1,  each shape == (num_frames, 4)
    trajs_2: list[tensor], len == n2,  each shape == (num_frames, 4)

    starts  shape == (n1,)
    starts_gt  shape == (n2,)
    '''
    trajs_1 = trajs
    trajs_2 = trajs_gt

    starts1 = starts
    starts2 = starts_gt

    assert isinstance(starts1,torch.Tensor) and isinstance(starts2,torch.Tensor)
    device = starts1.device
    n1,n2 = len(trajs_1),len(trajs_2)
    lens_1 = torch.as_tensor([traj.shape[0] for traj in trajs_1],device=device) # (n1,)
    lens_2 = torch.as_tensor([traj.shape[0] for traj in trajs_2],device=device) # (n2,)
    duras_1 = torch.stack([starts1,starts1+lens_1-1],dim=-1)  # (n1,2), closed interval
    duras_2 = torch.stack([starts2,starts2+lens_2-1],dim=-1)  # (n2,2)
    duras_inter,mask = dura_intersection_ts(duras_1,duras_2,broadcast=True) # (n1,n2,2), (n1,n2)
    vpois = torch.zeros_like(mask,dtype=torch.float)  # (n1,n2) == (n,n_gt)

    if broadcast:
        # duras_inter.shape==(n1,n2,2); mask.shape == (n1,n2)
        rel_duras_1 = duras_inter - duras_1[:,0,None,None]  # (n1,n2,2) # convert to relative duration
        rel_duras_2 = duras_inter - duras_2[None,:,0,None]  # (n1,n2,2)
        pos_ids_1,pos_ids_2 = mask.nonzero(as_tuple=True)  # row, col; positive ids
        for id1,id2 in zip(pos_ids_1.tolist(),pos_ids_2.tolist()):
            dura_1 = rel_duras_1[id1,id2,:]
            dura_2 = rel_duras_2[id1,id2,:]
            traj_1 = trajs_1[id1].float()   # (num_frames,4)
            traj_2 = trajs_2[id2].float()   # (num_frames,4)

            vpois[id1,id2] = _vPoI(traj_1,traj_2,dura_1,dura_2)
    else:
        assert n1==n2
        # duras_inter.shape==(n1,2); mask.shape == (n1,)
        rel_duras_1 = duras_inter - duras_1[:,0,None]  # (n1,2) # convert to relative duration
        rel_duras_2 = duras_inter - duras_2[:,0,None]  # (n1,2)
        pos_ids = mask.nonzero(as_tuple=True)[0]  # positive ids
        for idx in pos_ids:
            dura_1 = rel_duras_1[idx,:]
            dura_2 = rel_duras_2[idx,:]
            traj_1 = trajs_1[idx].float()   # (num_frames,4)
            traj_2 = trajs_2[idx].float()   # (num_frames,4)

            vpois[idx] = _vPoI(traj_1,traj_2,dura_1,dura_2)


    return vpois




def vIoU_aligned(traj_1,traj_2):

    assert isinstance(traj_1,torch.Tensor) and isinstance(traj_2,torch.Tensor)
    assert traj_1.shape == traj_2.shape
    
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)


def dura_intersection(dura1,dura2):
    s1,e1 = dura1
    assert s1 < e1 ,"dura1={},dura2={}".format(dura1,dura2)
    s2,e2 = dura2
    assert s2 < e2 ,"dura1={},dura2={}".format(dura1,dura2)
    if e1 <= s2 or e2 <= s1:        
        # because duration is [strat_fid, end_fid)   start_fid is inclusive and end_fid is exclusive
        # boundary points coinciding are not considered as intersection 
        return None
    
    inter_s = max(s1,s2)
    inter_e = min(e1,e2)

    return (inter_s, inter_e)

def traj_cutoff_close(ori_traj,ori_dura,dura,debug_info=None):
    """
    ori_traj: list[list], outside_len==num_frames,inside_len==4, or tensor of shape == (num_frames,4)
    ori_dura: list, or tensor of shape (2,), e.g., [23,33]
    dura:   e.g., [25,29]
    """
    assert len(ori_traj) == ori_dura[1] - ori_dura[0] + 1,"len(traj)={}!=end_fid-start_fid={},{}".format(len(ori_traj),ori_dura[1] - ori_dura[0],debug_info)
    s_o, e_o = ori_dura
    ss, ee = dura
    assert s_o <= ss and ee <= e_o,"ori_dura={},dura={},{}".format(ori_dura,dura,debug_info)

    index_s = ss - s_o
    index_e = index_s + (ee - ss)  # if index_s == index_e  then ori_traj[index_s:index_e] == []
    return ori_traj[index_s:index_e]

def traj_cutoff(ori_traj,ori_dura,dura,debug_info=None):
    """
    ori_traj: list[list], outside_len==num_frames,inside_len==4, or tensor of shape == (num_frames,x)
    ori_dura: tuple, e.g., (23,43)  # [start_fid,end_fid), end_fid is exclusive
    dura:   tuple,   e.g., (25,34)  # the same format as above
    """
    assert len(ori_traj) == ori_dura[1] - ori_dura[0],"len(traj)={}!=end_fid-start_fid={},{}".format(len(ori_traj),ori_dura[1] - ori_dura[0],debug_info)
    s_o, e_o = ori_dura
    ss, ee = dura
    assert s_o <= ss and ee <= e_o,"ori_dura={},dura={},{}".format(ori_dura,dura,debug_info)

    index_s = ss - s_o
    index_e = len(ori_traj) - (e_o - ee)
    return ori_traj[index_s:index_e]

def vIoU(traj_1, duration_1, traj_2, duration_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
    assert type(traj_1) == type(traj_2), "{}, {}".format(type(traj_1),type(traj_2))
    if isinstance(traj_1,torch.Tensor):
        traj_1 = traj_1.float()
        traj_2 = traj_2.float()
    elif isinstance(traj_1,np.ndarray):
        traj_1 = traj_1.astype(np.float32)
        traj_2 = traj_2.astype(np.float32)
    else:
        assert isinstance(traj_1,list)

    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.0
    elif duration_1[0] <= duration_2[0]:
        head_1 = duration_2[0] - duration_1[0]
        head_2 = 0
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    else:
        head_1 = 0
        head_2 = duration_1[0] - duration_2[0]
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    v_overlap = 0
    for i in range(tail_1 - head_1):
        roi_1 = traj_1[head_1 + i]
        roi_2 = traj_2[head_2 + i]
        left = max(roi_1[0], roi_2[0])
        top = max(roi_1[1], roi_2[1])
        right = min(roi_1[2], roi_2[2])
        bottom = min(roi_1[3], roi_2[3])
        v_overlap += max(0, right - left + 1) * max(0, bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += (traj_1[i][2] - traj_1[i][0] + 1) * (traj_1[i][3] - traj_1[i][1] + 1)
    v2 = 0
    for i in range(len(traj_2)):
        v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)

def merge_duration_list(duration_list):
    """
    在vidvrd中，会有一个连续60帧的predicate，被分别标注为3个30帧的predicate (overlap 15帧)
    e.g., input: duration_list == [(195, 225), (210, 240), (225, 255), (240, 270),
        (255, 285), (375, 405), (390, 420), (405, 435),
        (645, 675), (660, 690), (675, 705), (690, 720), 
        (705, 735), (720, 750), (780, 810), (795, 825), (810, 840), (825, 855)]
        
        return: merged_durations == [(195, 285), (375, 435), (645, 750), (780, 855)]
    """
    # print("duration_list:",duration_list)
    duration_list = duration_list.copy()
    duration_list = sorted(duration_list,key=lambda d: d[0])  # 从小到大排序
    merged_durations = []
    head_dura = duration_list.pop(0)
    merged_durations.append(head_dura)

    while duration_list != []:
        former_dura = merged_durations[-1]
        former_start,former_end = former_dura

        cur_dura = duration_list.pop(0)
        cur_start,cur_end = cur_dura
        if cur_start <= former_end:
            merged_durations.pop(-1)
            merged_dura = (former_start,cur_end)
            merged_durations.append(merged_dura)
        else:
            merged_durations.append(cur_dura)
    # print("after merge:",merged_durations)
    return merged_durations

def is_overlap_old(dura1,dura2):
    dura_list = [dura1,dura2]
    dura_list = dura_list.copy()
    dura_list = sorted(dura_list,key=lambda d: d[0]) # 升序
    d1_start,d1_end = dura_list[0]
    assert d1_start <= d1_end
    d2_start,d2_end = dura_list[1]
    assert d2_start <= d2_end

    if d2_start < d1_end:
        return True
    else:
        return False

def is_overlap(dura1,dura2):
    s1,e1 = dura1
    assert s1 < e1 
    s2,e2 = dura2
    assert s2 < e2

    if e1 <= s2 or e2 <= s1:
        # because duration is [strat_fid, end_fid)   start_fid is inclusive and end_fid is exclusive
        # boundary points coinciding are not considered as intersection 
        return False
    else:
        return True

def temporal_overlap(dura1,dura2):
    s1,e1 = dura1
    assert s1 < e1 
    s2,e2 = dura2
    assert s2 < e2

    overlap_len = min(e1,e2) - max(s1,s2)
    return overlap_len



def collator_func_v1(batch):
    """
    batch is a list ,len(batch) == batch_size
    batch[i] is a tuple, batch[i][0],batch[i][1] is an object of class TrajProposal, class VideoGraph, respectively
    This function should be passed to the torch.utils.data.DataLoader

    return:

    """
    batch_size = len(batch)
    batch_proposal = [b[0] for b in batch]
    batch_gt_graph = [b[1] for b in batch]

    # process of proposals
    proposal_num_list = [b[1].num_proposals for b in batch]
    max_n_proposal = max(proposal_num_list)
    batch_cat_ids = []
    batch_traj_boxes = []
    batch_durations = []
    batch_roi_features = []
    for traj in batch_proposal:
        batch_cat_ids.append(traj.cat_ids)
        batch_traj_boxes.append(traj.traj_boxes)
        batch_durations.append(traj.traj_durations)
        
        n_p,dim_feat = traj.roi_features.shape               # shape == (num_proposals, dim_feat)
        after_padding = np.zeros(shape=(max_n_proposal,dim_feat))
        after_padding[:n_p,:] = traj.roi_features
        batch_roi_features.append(after_padding)     
    batch_roi_features = np.stack(batch_roi_features,axis=0) # shape == (batch_size,max_n_proposal,dim_feat) 

    proposal_dict = {
        "proposal_num_list":proposal_num_list,  # list[int], len==batch_size
        "cat_ids":batch_cat_ids,                # list[list[int]], outside_len==batch_size,inside_len==num_proposals
        "traj_boxes":batch_traj_boxes,          # list[list[np.ndarray]], outside_len==batch_size,inside_len==num_proposals, np.ndarray.shape==(num_frames,4) #TODO consider zeropadding
        "durations":batch_durations,            # list[list[tuple]],outside_len==batch_size,inside_len==num_proposals, tuple==(start_framd_id,end_frame_id)
        "roi_features":batch_roi_features       # np.ndarray, shape == (batch_size,max_n_proposal,dim_feat), with zero padding
    }

    # process of gt_graphs
    batch_traj_cat_ids = []
    batch_traj_durations = []
    batch_traj_bboxes = []

    batch_pred_cat_ids = []
    batch_pred_durations = []
    for graph in batch_gt_graph:
        batch_traj_cat_ids.append(graph.traj_cat_ids)
        batch_pred_cat_ids.append(graph.pred_cat_ids)
        # TODO 未完待续
        # self. = traj_cat_ids
        # self. = traj_durations
        # self. = traj_bboxes

        # self.pred_cat_ids = pred_cat_ids
        # self. = pred_durations

        # assert adj_matrix_object.shape == adj_matrix_subject.shape
        # self.adj_matrix = np.stack([adj_matrix_subject,adj_matrix_object],axis=0)   # shape = (2,num_preds,num_trajs)
    
    return batch


def collator_func_v2(batch):
    """
    batch is a list ,len(batch) == batch_size
    batch[i] is a tuple, batch[i][0],batch[i][1] is an object of class TrajProposal, class VideoGraph, respectively
    This function should be passed to the torch.utils.data.DataLoader

    """
    # batch_size = len(batch)
    batch_proposal = [b[0] for b in batch]
    batch_gt_graph = [b[1] for b in batch]

    return batch_proposal,batch_gt_graph


def collator_func_sort(batch):
    """
    batch is a list ,len(batch) == batch_size
    batch[i] is a tuple, batch[i][0],batch[i][1] is an object of class TrajProposal, class VideoGraph, respectively
    This function should be passed to the torch.utils.data.DataLoader

    """
    # batch_size = len(batch)
    batch = sorted(batch,key=lambda x: x[1].max_frames)

    batch_proposal = [b[0] for b in batch]
    batch_gt_graph = [b[1] for b in batch]

    return batch_proposal,batch_gt_graph

def collator_func_gt(batch):
    """
    batch is a list ,len(batch) == batch_size
    batch[i] is a tuple, batch[i][0],batch[i][1] is an object of class TrajProposal, class VideoGraph, respectively
    This function should be passed to the torch.utils.data.DataLoader

    """
    # batch_size = len(batch)

    batch_proposal = [b[0] for b in batch]
    batch_gt_graph = [b[1] for b in batch]
    batch_gt_feature = [b[2] for b in batch]

    return batch_proposal,batch_gt_graph,batch_gt_feature


def collator_func_v3(batch):
    """
    batch is a list ,len(batch) == batch_size
    batch[i] is a tuple, batch[i][0],batch[i][1] is an object of class TrajProposal, class VideoGraph, respectively
    This function should be passed to the torch.utils.data.DataLoader

    """

    batch_video_features = [b[0] for b in batch]
    batch_video_features = torch.stack(batch_video_features,dim=0)
    batch_proposal = [b[1] for b in batch]
    batch_gt_graph = [b[2] for b in batch]

    return batch_video_features,batch_proposal,batch_gt_graph