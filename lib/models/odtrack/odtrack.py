import math
import os
from typing import List
import collections
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.odtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.odtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xywh_to_cxcywh
# MODIFICATION START
from .motion_head import StateSpaceHead, create_gaussian_attention_bias
# MODIFICATION END

class ODTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1,motion_head_enable=False, motion_head_hidden_dim=128,
                 # MODIFICATION START
                 motion_history_len=3):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

        # MODIFICATION START
        self.motion_head_enable = motion_head_enable
        if self.motion_head_enable:
            self.history_len = motion_history_len
            self.motion_head = StateSpaceHead(history_len=self.history_len,hidden_dim=motion_head_hidden_dim)
            # We need to store the previous box to feed to the motion model
            self.history_boxes_cxcywh = collections.deque(maxlen=self.history_len)

    def reset_motion_history(self):
            """ Resets the motion model's history. Should be called at the start of each new video sequence. """
            if self.motion_head_enable:
                self.history_boxes_cxcywh.clear()
     
        # MODIFICATION END

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                gt_bboxes_xywh: List[torch.Tensor] = None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"
        if self.training:
             # During training, we process one sequence clip at a time, so reset every time.
            self.reset_motion_history()
        out_dict = []
        for i in range(len(search)):
            # MODIFICATION START - Motion Prediction
            attention_bias = None
            motion_pred_delta = None
            motion_pred_log_sigma = None
            
            if self.motion_head_enable and len(self.history_boxes_cxcywh) >0:
                # 1. Predict motion delta and uncertainty
                current_history = list(self.history_boxes_cxcywh)
                if len(current_history) < self.history_len:
                    # Repeat the oldest available box (the first one) to pad.
                    num_padding = self.history_len - len(current_history)
                    padding = [current_history[0]] * num_padding
                    padded_history = padding + current_history
                else:
                    padded_history = current_history
                history_tensor = torch.stack(padded_history, dim=1) # (B, H_len, 4)
                history_flat = history_tensor.flatten(start_dim=1) # (B, H_len * 4)
                motion_pred_delta, motion_pred_log_sigma = self.motion_head(history_flat)
                last_box = self.history_boxes_cxcywh[-1]
                # 2. Predict next box center
                predicted_box_cxcywh = last_box+ motion_pred_delta
                predicted_center_xy = predicted_box_cxcywh[:, :2] # Normalized (cx, cy)
                
                # 3. Create attention bias map
                attention_bias = create_gaussian_attention_bias(
                    predicted_center_xy, self.feat_sz_s, search[i].device
                )
            # MODIFICATION END
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len,attention_bias=attention_bias)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            # MODIFICATION START - Update history and add motion predictions to output
            if self.training:
                # Use ground truth to update history for teacher-forcing
                current_gt_box_xywh = gt_bboxes_xywh[i]
                current_box_cxcywh = box_xywh_to_cxcywh(current_gt_box_xywh.clone())
                #self.history_box_cxcywh = box_xywh_to_cxcywh(current_gt_box_xywh.clone())
            else:
                # During inference, use the model's own prediction
                current_box_cxcywh = out['pred_boxes'].view(-1, 4).detach()
                #self.history_box_cxcywh = out['pred_boxes'].view(-1, 4).detach()
            if self.motion_head_enable:
                self.history_boxes_cxcywh.append(current_box_cxcywh)
            if motion_pred_delta is not None:
                out['motion_pred_delta'] = motion_pred_delta
                out['motion_pred_log_sigma'] = motion_pred_log_sigma
            # MODIFICATION END

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_odtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ODTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
        motion_head_enable=cfg.MODEL.MOTION.ENABLE,
        motion_head_hidden_dim=cfg.MODEL.MOTION.HEAD_HIDDEN_DIM,
        motion_history_len=cfg.MODEL.MOTION.HISTORY_LEN,
    )

    return model
