import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU


class PAN_PP(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(PAN_PP, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.rec_head = None
        if recognition_head:
            self.rec_head = build_head(recognition_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                img_metas=None,
                cfg=None):
        if cfg.debug:
            from IPython import embed
            embed()

        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)

        # FFM
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        out_det = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))
            start = time.time()

        if self.training:
            out_det = self._upsample(out_det, imgs.size())
            loss_det = self.det_head.loss(
                out_det, gt_texts, gt_kernels, training_masks,
                gt_instances, gt_bboxes)
            outputs.update(loss_det)
        else:
            out_det = self._upsample(out_det, imgs.size(), cfg.test_cfg.scale)
            res_det = self.det_head.get_results(out_det, img_metas, cfg)
            outputs.update(res_det)

        if self.rec_head is not None:
            if self.training:
                x_crops, gt_words = self.rec_head.extract_feature(
                    f, (imgs.size(2), imgs.size(3)),
                    gt_instances * training_masks, gt_bboxes, gt_words,
                    word_masks)

                if x_crops is not None:
                    out_rec = self.rec_head(x_crops, gt_words)
                    loss_rec = self.rec_head.loss(out_rec, gt_words,
                                                  reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1,), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1,), -1, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
            else:
                if len(res_det['bboxes']) > 0:
                    x_crops, _ = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(res_det['label'],
                                     dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(res_det['bboxes_h'],
                                            dtype=torch.long),
                        unique_labels=res_det['instances'])
                    words, word_scores = self.rec_head.forward(x_crops)
                else:
                    words = []
                    word_scores = []

                if cfg.report_speed:
                    torch.cuda.synchronize()
                    outputs.update(dict(rec_time=time.time() - start))
                outputs.update(
                    dict(words=words, word_scores=word_scores, label=''))

        return outputs
