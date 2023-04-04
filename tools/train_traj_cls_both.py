
import argparse
import os 
from tqdm import tqdm
from collections import defaultdict

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.TrajClsModel_v2 import OpenVocTrajCls as OpenVocTrajCls_NoBgEmb
from models.TrajClsModel_v3 import OpenVocTrajCls as OpenVocTrajCls_0BgEmb
# from dataloaders.dataset_vidor_v2 import VidORTrajDataset
from dataloaders.dataset_vidor_v3 import VidORTrajDataset
from dataloaders.dataset_vidvrd_v3 import VidVRDTrajDataset
from utils.utils_func import get_to_device_func
from utils.config_parser import parse_config_py
from utils.logger import LOGGER, add_log_to_file

def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size

def save_checkpoint(batch_size,crt_epoch,model,optimizer,scheduler,save_path):
    checkpoint = {
        "batch_size":batch_size,
        "crt_epoch":crt_epoch + 1,
        "model_state_dict":model.state_dict(),
        "optim_state_dict":optimizer.state_dict(),
        "sched_state_dict":scheduler.state_dict(),
    }
    torch.save(checkpoint,save_path)




def train_TrajClsOpenVoc_with_eval(model_class,cfg_path,output_dir=None, use_distillation=True,from_checkpoint = False,ckpt_path = None,save_tag=""):
    ########## TODO:
    ####### writing here
    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_log_dir = os.path.join(log_dir,"tensorboard_{}/".format(save_tag))
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    if not from_checkpoint:
        os.system("rm {}events*".format(tensorboard_log_dir))
    writer = SummaryWriter(tensorboard_log_dir)

    log_path = os.path.join(log_dir,f'train_{save_tag}.log')
    add_log_to_file(log_path)
    

    configs = parse_config_py(cfg_path)
    dataset_cfg = configs["train_dataset_cfg"]
    model_cfg = configs["model_cfg"]
    train_cfg = configs["train_cfg"]
    device = torch.device("cuda")

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("training config: {}".format(train_cfg))

    batch_size          = train_cfg["batch_size"]
    total_epoch         = train_cfg["total_epoch"]
    initial_lr          = train_cfg["initial_lr"]
    lr_decay            = train_cfg["lr_decay"]
    epoch_lr_milestones = train_cfg["epoch_lr_milestones"]

    model = model_class(model_cfg,is_train=True,use_distillation=use_distillation)
    model = model.to(device)
    model.reset_classifier_weights("base")
    
    LOGGER.info("preparing dataloader...")
    dataset = VidORTrajDataset(**dataset_cfg)
    collate_func = dataset.get_collator_func()
    to_device_func = get_to_device_func(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn = collate_func ,
        num_workers = 12,
        drop_last= False,
        shuffle= True,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "len(dataset)=={},batch_size=={},len(dataloader)=={},{}x{}={}".format(
            dataset_len,batch_size,dataloader_len,batch_size,dataloader_len,batch_size*dataloader_len
        )
    )

    milestones = [int(m*dataloader_len) for m in epoch_lr_milestones]
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=lr_decay)


    if from_checkpoint:
        model,optimizer,scheduler,crt_epoch,batch_size_ = load_checkpoint(model,optimizer,scheduler,ckpt_path)
        # assert batch_size == batch_size_ , "batch_size from checkpoint not match : {} != {}"
        if batch_size != batch_size_:
            LOGGER.warning(
                "!!!Warning!!! batch_size from checkpoint not match : {} != {}".format(batch_size,batch_size_)
            )
        LOGGER.info("checkpoint load from {}".format(ckpt_path))
    else:
        crt_epoch = 0

    LOGGER.info("start training:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("use device: {}".format(device))
    LOGGER.info("weights will be saved in output_dir = {}".format(output_dir))


    it=0
    for epoch in tqdm(range(total_epoch)):
        if epoch < crt_epoch:
            it+=dataloader_len
            continue

        epoch_loss = defaultdict(list)
        for batch_data in dataloader:
            (
                video_names,
                batch_traj_infos,
                batch_traj_feats,
                bacth_traj_embds,
                batch_gt_annos,
                batch_labels
            ) = batch_data

            input_data = (
                batch_traj_feats,
                bacth_traj_embds,
                batch_labels
            )
            input_data = tuple(to_device_func(data) for data in input_data)

            optimizer.zero_grad()
            total_loss, loss_dict = model(input_data)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            scheduler.step()


            loss_str = "epoch={};iter={}; ".format(epoch,it)
            for k,v in loss_dict.items():
                epoch_loss[k].append(v.item())
                loss_str += "{}:{:.4f}; ".format(k,v.item())
                writer.add_scalar('Iter/{}'.format(k), v.item(), it)
            loss_str += "lr={}".format(optimizer.param_groups[0]["lr"])
            if it % 20 == 0:
                LOGGER.info(loss_str)
            it+=1
    
        epoch_loss_str = "mean_loss_epoch={}: ".format(epoch)
        for k,v in epoch_loss.items():
            v = np.mean(v)
            writer.add_scalar('Epoch/{}'.format(k), v, epoch)
            epoch_loss_str += "{}:{:.4f}; ".format(k,v)
        LOGGER.info(epoch_loss_str)
        
        if epoch >0 and epoch % 10 == 0:
            save_path = os.path.join(output_dir,'model_{}_epoch_{}.pth'.format(save_tag,epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            LOGGER.info("checkpoint is saved: {}".format(save_path))
    
    save_path = os.path.join(output_dir,'model_final_{}_epoch_{}.pth'.format(save_tag,total_epoch))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    LOGGER.info("checkpoint is saved: {}".format(save_path))
    LOGGER.handlers.clear()


def train_TrajClsOpenVoc(model_class,dataset_class,args):
    cfg_path = args.cfg_path
    output_dir = args.output_dir
    use_distillation = args.use_distillation
    from_checkpoint = args.from_checkpoint
    ckpt_path = args.ckpt_path
    save_tag = args.save_tag

    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_log_dir = os.path.join(log_dir,"tensorboard_{}/".format(save_tag))
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    if not from_checkpoint:
        os.system("rm {}events*".format(tensorboard_log_dir))
    writer = SummaryWriter(tensorboard_log_dir)

    log_path = os.path.join(log_dir,f'train_{save_tag}.log')
    add_log_to_file(log_path)
    

    configs = parse_config_py(cfg_path)
    dataset_cfg = configs["train_dataset_cfg"]
    model_cfg = configs["model_cfg"]
    train_cfg = configs["train_cfg"]
    device = torch.device("cuda")

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("training config: {}".format(train_cfg))

    batch_size          = train_cfg["batch_size"]
    total_epoch         = train_cfg["total_epoch"]
    initial_lr          = train_cfg["initial_lr"]
    lr_decay            = train_cfg["lr_decay"]
    epoch_lr_milestones = train_cfg["epoch_lr_milestones"]

    model = model_class(model_cfg,is_train=True,use_distillation=use_distillation)
    model = model.to(device)
    model.reset_classifier_weights("base")
    
    LOGGER.info("preparing dataloader...")
    dataset = dataset_class(**dataset_cfg)
    collate_func = dataset.get_collator_func()
    to_device_func = get_to_device_func(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn = collate_func ,
        num_workers = 12,
        drop_last= False,
        shuffle= True,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "len(dataset)=={},batch_size=={},len(dataloader)=={},{}x{}={}".format(
            dataset_len,batch_size,dataloader_len,batch_size,dataloader_len,batch_size*dataloader_len
        )
    )

    milestones = [int(m*dataloader_len) for m in epoch_lr_milestones]
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=lr_decay)


    if from_checkpoint:
        model,optimizer,scheduler,crt_epoch,batch_size_ = load_checkpoint(model,optimizer,scheduler,ckpt_path)
        # assert batch_size == batch_size_ , "batch_size from checkpoint not match : {} != {}"
        if batch_size != batch_size_:
            LOGGER.warning(
                "!!!Warning!!! batch_size from checkpoint not match : {} != {}".format(batch_size,batch_size_)
            )
        LOGGER.info("checkpoint load from {}".format(ckpt_path))
    else:
        crt_epoch = 0

    LOGGER.info("start training:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("use device: {}".format(device))
    LOGGER.info("weights will be saved in output_dir = {}".format(output_dir))


    it=0
    for epoch in tqdm(range(total_epoch)):
        if epoch < crt_epoch:
            it+=dataloader_len
            continue

        epoch_loss = defaultdict(list)
        for batch_data in dataloader:
            (
                video_names,
                batch_traj_infos,
                batch_traj_feats,
                bacth_traj_embds,
                batch_gt_annos,
                batch_labels
            ) = batch_data

            input_data = (
                batch_traj_feats,
                bacth_traj_embds,
                batch_labels
            )
            input_data = tuple(to_device_func(data) for data in input_data)

            optimizer.zero_grad()
            total_loss, loss_dict = model(input_data)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            scheduler.step()


            loss_str = "epoch={};iter={}; ".format(epoch,it)
            for k,v in loss_dict.items():
                epoch_loss[k].append(v.item())
                loss_str += "{}:{:.4f}; ".format(k,v.item())
                writer.add_scalar('Iter/{}'.format(k), v.item(), it)
            loss_str += "lr={}".format(optimizer.param_groups[0]["lr"])
            if it % 20 == 0:
                LOGGER.info(loss_str)
            it+=1
    
        epoch_loss_str = "mean_loss_epoch={}: ".format(epoch)
        for k,v in epoch_loss.items():
            v = np.mean(v)
            writer.add_scalar('Epoch/{}'.format(k), v, epoch)
            epoch_loss_str += "{}:{:.4f}; ".format(k,v)
        LOGGER.info(epoch_loss_str)
        
        if epoch >0 and epoch % 10 == 0:
            save_path = os.path.join(output_dir,'model_{}_epoch_{}.pth'.format(save_tag,epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            LOGGER.info("checkpoint is saved: {}".format(save_path))
    
    save_path = os.path.join(output_dir,'model_final_{}_epoch_{}.pth'.format(save_tag,total_epoch))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    LOGGER.info("checkpoint is saved: {}".format(save_path))
    LOGGER.handlers.clear()



if __name__ == "__main__":
    random.seed(111)
    np.random.seed(111)
    torch.random.manual_seed(111)


    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--model_class", type=str,help="...")
    parser.add_argument("--dataset_class", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="default `output_dir` will be set as the dir of `cfg_path`")
    parser.add_argument("--use_distillation", action="store_true")  # 
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--train_baseline", action="store_true")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--save_tag", type=str,default="")
    parser.add_argument("--other_cfgs",nargs="+",default=[])
  
    args = parser.parse_args()


    # for cfg in cfgs:
    model_class = eval(args.model_class)
    dataset_class = eval(args.dataset_class)

    if args.train_baseline:
        pass  # TODO
    else:
        save_tag_ = "with_distil" if args.use_distillation else "wo_distil"
        args.save_tag = save_tag_ + '_' + args.save_tag
        train_TrajClsOpenVoc(
            model_class,
            dataset_class,
            args
        )

    # OpenVocTrajCls_0BgEmb
    # OpenVocTrajCls_NoBgEmb
    # VidVRDTrajDataset
    '''
    !!!!!! NOTE export the path environment variable first
    export PYTHONPATH=$PYTHONPATH:"/your_project_path/" (e.g., "/home/username/OpenVoc-VidVRD")

    ########## VidVRD

    CUDA_VISIBLE_DEVICES=1 python tools/train_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_NoBgEmb \
        --cfg_path experiments_vidvrd_trajcls/OpenVocTrajCls_NoBgEmb/cfg_trajcls.py \
        --output_dir experiments_vidvrd_trajcls/OpenVocTrajCls_NoBgEmb/ \
        --use_distillation \
        --save_tag w5bs128
    
    CUDA_VISIBLE_DEVICES=3 python tools/train_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_NoBgEmb \
        --cfg_path experiments_vidvrd_trajcls/OpenVocTrajCls_NoBgEmb/cfg_trajcls.py \
        --output_dir experiments_vidvrd_trajcls/OpenVocTrajCls_NoBgEmb/ \
        --save_tag bs128

    
    CUDA_VISIBLE_DEVICES=2 python tools/train_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path experiments_vidvrd_trajcls/OpenVocTrajCls_0BgEmb/cfg_trajcls.py \
        --output_dir experiments_vidvrd_trajcls/OpenVocTrajCls_0BgEmb/ \
        --use_distillation \
        --save_tag w5bs128
    
    CUDA_VISIBLE_DEVICES=3 python tools/train_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path experiments_vidvrd_trajcls/OpenVocTrajCls_0BgEmb/cfg_trajcls.py \
        --output_dir experiments_vidvrd_trajcls/OpenVocTrajCls_0BgEmb/ \
        --save_tag bs128



    ######### VidOR
    CUDA_VISIBLE_DEVICES=1 python tools/train_traj_cls_both.py \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path experiments_vidor/OpenVocTrajCls_0BgEmb/cfg_trajcls.py \
        --output_dir experiments_vidor/OpenVocTrajCls_0BgEmb/ \
        --use_distillation \
        --save_tag w5bs16
    
    CUDA_VISIBLE_DEVICES=1 python tools/train_traj_cls_both.py \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path experiments_vidor/OpenVocTrajCls_0BgEmb/cfg_trajcls.py \
        --output_dir experiments_vidor/OpenVocTrajCls_0BgEmb/ \
        --save_tag bs16
    
    ###
    CUDA_VISIBLE_DEVICES=3 python tools/train_traj_cls_both.py \
        --model_class OpenVocTrajCls_NoBgEmb \
        --cfg_path experiments_vidor/OpenVocTrajCls_NoBgEmb/cfg_trajcls.py \
        --output_dir experiments_vidor/OpenVocTrajCls_NoBgEmb/ \
        --use_distillation \
        --save_tag w5bs16
    
    ###
    CUDA_VISIBLE_DEVICES=3 python tools/train_traj_cls_both.py \
        --model_class OpenVocTrajCls_NoBgEmb \
        --cfg_path experiments_vidor/OpenVocTrajCls_NoBgEmb/cfg_trajcls.py \
        --output_dir experiments_vidor/OpenVocTrajCls_NoBgEmb/ \
        --save_tag bs16
    
    
   
    
    export TMPDIR=/tmp/$USER 
    tensorboard --logdir=/home/gkf/project/VidVRD-OpenVoc/experiments_vidor/TrajCls_video_lvl_data/logfile/tensorboard_OpenVoc_w5bs16 --port=6006 --bind_all

tensorboard --logdir_spec=\
with_distil:/home/gkf/project/VidVRD-OpenVoc/experiments_vidor/TrajCls_video_lvl_data/logfile/tensorboard_OpenVoc_w5bs16,\
wo_distil:/home/gkf/project/VidVRD-OpenVoc/experiments_vidor/TrajCls_video_lvl_data/logfile/tensorboard_ZeroShot_bs16\
  --port=6010 --bind_all
  
    '''