import torch
import torch.nn as nn
import torch.nn.functional as F
from moco import GCN_Transformer
from moco import GCN_Transformer_mask
from moco import decoder


def init_para_GCN_Trans(model):
    print("<<< initialize the para in transformer!")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class Body2DLoss:
    def __init__(self, lamda_body):
        self.lamda = lamda_body
        self.loss_func = torch.nn.L1Loss(reduce=False).cuda()

    def compute_loss(self, pred, targets, flag_2d, scale, mask):
        scale = scale
        if flag_2d is None:
            flag_2d = 1.0
        b = self.loss_func(targets, pred) * flag_2d
        if mask is None:
            loss_2d = self.lamda * torch.mean(b)
        else:
            loss_2d = self.lamda * torch.sum(b* mask.unsqueeze(-1)) / (torch.sum(mask)* pred.shape[2] + 1.0)
        return loss_2d


class joint2Dloss:
    def __init__(self, lamda_kp2d):
        self.lamda = lamda_kp2d
        self.loss_func = torch.nn.L1Loss(reduce=False).cuda()

    def compute_loss(self, pred, targets, flag_2d, scale, mask):
        scale = scale
        if flag_2d is None:
            flag_2d = 1.0
        b = self.loss_func(targets, pred) * flag_2d
        if mask is None:
            loss_2d = self.lamda * torch.mean(b)
        else:
            loss_2d = self.lamda * torch.sum(b * mask.unsqueeze(-1)) / (torch.sum(mask) * pred.shape[2] + 1.0)
        return loss_2d


class MASA(nn.Module):
    def __init__(self, skeleton_representation, num_class, dim=128, K=65536, m=0.999, T=0.07,
                 teacher_T=0.05, student_T=0.1, topk=1024, mlp=False, pretrain=True, dropout=None,
                 inter_weight=0.5, inter_dist=False, topk_part=8192, K_part=16384):
        super(MASA, self).__init__()
        self.pretrain = pretrain
        RHand_Bone = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 1), (7, 6), (8, 7), (9, 8), (10, 1),
                      (11, 10), (12, 11), (13, 12), (14, 1), (15, 14), (16, 15), (17, 16), (18, 1),
                      (19, 18), (20, 19), (21, 20)]
        Body_Bone = [(2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6)]
        self.Bone = RHand_Bone + [(k + 21, v + 21) for k, v in RHand_Bone] + [(k + 42, v + 42) for k, v in Body_Bone]
        if not self.pretrain:
            cfg = GCN_Transformer.Config()
            cfg.proj_dropout = dropout
            cfg.num_class = num_class
            self.encoder_q = GCN_Transformer.Model(cfg)
        else:
            self.K = K
            self.m = m
            self.T = T
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.inter_weight = inter_weight
            self.topk = topk
            self.K_part = K_part
            self.topk_part = topk_part
            mlp = mlp
            print(skeleton_representation)

            cfg_q = GCN_Transformer_mask.Config()
            cfg_q.proj_dropout = dropout
            cfg_q.inter_dist = inter_dist
            cfg_k = GCN_Transformer_mask.Config()
            cfg_k.proj_dropout = dropout
            cfg_k.inter_dist = inter_dist
            self.encoder_q = GCN_Transformer_mask.Model(cfg_q)
            self.encoder_k = GCN_Transformer_mask.Model(cfg_k)
            init_para_GCN_Trans(self.encoder_q)
            init_para_GCN_Trans(self.encoder_k)

            # projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.proj.fc.weight.shape[1]
                self.encoder_q.proj.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q.proj.fc)
                self.encoder_k.proj.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k.proj.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        #create the decoder for regression
        self.body_decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.ReLU(), 
        )
        self.hand_decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 42),
            torch.nn.ReLU(),
        )
        self.target_decoder = decoder.Decoder(decoder.Config())
        self.reg_decoder = decoder.Decoder(decoder.Config())
        self.hand_loss = joint2Dloss(lamda_kp2d=10.0)
        self.body_loss = Body2DLoss(lamda_body=10.0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def pose_process(self, feat):
        B, T, C = feat.size()
        rh_feat = feat[:, :, :512]
        lh_feat = feat[:, :, 512:1024]
        body_feat = feat[:, :, 1024:1536]

        rh_pose = self.hand_decoder(rh_feat).view(B, T, 21, 2)
        lh_pose = self.hand_decoder(lh_feat).view(B, T, 21, 2)
        body_pose = self.body_decoder(body_feat).view(B, T, 7, 2)
        return rh_pose, lh_pose, body_pose



    def forward(self, im_q, im_k=None, view='joint', knn_eval=False, self_dist=False):
        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)[0]
            else:
                raise ValueError


        # compute query features
        feat, enc_mask = self.encoder_q(im_q)  # queries: NxC
        q, q_hand, q_body = self.encoder_q.predict_head(feat, im_q, enc_mask=enc_mask)
        q = F.normalize(q, dim=1)


        # compute key features for  s1 and  s2  skeleton representations
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_k, enc_mask_k = self.encoder_k(im_k)  # keys: NxC
            k, k_hand, k_body = self.encoder_k.predict_head(feat_k, im_k, enc_mask=enc_mask_k)
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)


        # regression
        dec_feat = self.reg_decoder(feat, im_q['right']['vid_len'])
        rh_pose, lh_pose, body_pose = self.pose_process(dec_feat)
        rh_loss = self.hand_loss.compute_loss(rh_pose, im_q['right']['gts'], im_q['right']['flag_2d'], 256.0, im_q['right']['masked'])
        lh_loss = self.hand_loss.compute_loss(lh_pose, im_q['left']['gts'], im_q['left']['flag_2d'], 256.0, im_q['left']['masked'])
        body_loss = self.body_loss.compute_loss(body_pose, im_q['body']['body_pose_gt'], im_q['body']['body_pose_conf'], 256.0, im_q['body']['masked'])

        return logits, labels, rh_loss, lh_loss, body_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output