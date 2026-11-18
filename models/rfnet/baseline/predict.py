import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc

cudnn.benchmark = True

path = os.path.dirname(__file__)
from utils.generate import generate_snapshot


# def softmax_output_dice_class4(output, target):
#     eps = 1e-8
#     #######label1########
#     o1 = (output == 1).float()
#     t1 = (target == 1).float()
#     intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
#     denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
#     ncr_net_dice = intersect1 / denominator1

#     o2 = (output == 2).float()
#     t2 = (target == 2).float()
#     intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
#     denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
#     edema_dice = intersect2 / denominator2

#     o3 = (output == 3).float()
#     t3 = (target == 3).float()
#     intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
#     denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
#     enhancing_dice = intersect3 / denominator3

#     ####post processing:
#     if torch.sum(o3) < 500:
#        o4 = o3 * 0.0
#     else:
#        o4 = o3
#     t4 = t3
#     intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
#     denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
#     enhancing_dice_postpro = intersect4 / denominator4

#     o_whole = o1 + o2 + o3 
#     t_whole = t1 + t2 + t3 
#     intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
#     denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
#     dice_whole = intersect_whole / denominator_whole

#     o_core = o1 + o3
#     t_core = t1 + t3
#     intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
#     denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
#     dice_core = intersect_core / denominator_core

#     dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
#     dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

#     return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()



def softmax_output_dice_class4(output, target):
    """
    Compute per‐label and composite Dice exactly as in the official BraTS2023 "full" Dice:
      - Dice = 1.0 if both P and G are empty
      - otherwise  Dice = 2|P∩G| / (|P| + |G|)
    Keeps the original post‐processing on ET clusters < 500 voxels.
    
    Args:
        output (LongTensor): shape (B,H,W,D), integer labels {0,1,2,3}
        target (LongTensor): same shape/format as output
    Returns:
        dice_separate: (B,3) array [NCR, ED, ET]
        dice_evaluate: (B,4) array [WT, TC, ET, ET_postproc]
    """
    eps = 1e-6

    # helper to compute one region’s Dice with BraTS empty‐empty=1 rule
    def _brats_dice(o, t):
        vp = o.sum(dim=(1,2,3))
        vg = t.sum(dim=(1,2,3))
        inter = (o * t).sum(dim=(1,2,3))
        # empty‐empty → 1.0; else 2*inter/(vp+vg+eps)
        return torch.where(
            (vp + vg) == 0,
            torch.ones_like(vp, dtype=torch.float32),
            2.0 * inter / (vp + vg + eps)
        )

    # per‐label predictions
    o1, t1 = (output == 1).float(), (target == 1).float()  # NCR (non‐enhancing core)
    o2, t2 = (output == 2).float(), (target == 2).float()  # ED (edema)
    o3, t3 = (output == 3).float(), (target == 3).float()  # ET (enhancing tumor)

    ncr_net_dice   = _brats_dice(o1, t1)
    edema_dice     = _brats_dice(o2, t2)
    enhancing_dice = _brats_dice(o3, t3)

    # post‐processing: remove tiny ET clusters
    # (original logic used `if torch.sum(o3)<500`, assumes B=1)
    # Here we broadcast the threshold per‐sample:
    vp3 = o3.sum(dim=(1,2,3))
    mask_keep = (vp3 >= 500).float().view(-1,1,1,1)
    o4 = o3 * mask_keep
    enhancing_dice_postpro = _brats_dice(o4, t3)

    # composite regions
    o_whole = (o1 + o2 + o3).clamp(0,1)
    t_whole = (t1 + t2 + t3).clamp(0,1)
    dice_whole = _brats_dice(o_whole, t_whole)

    o_core = (o1 + o3).clamp(0,1)  # tumor core = NCR + ET
    t_core = (t1 + t3).clamp(0,1)
    dice_core = _brats_dice(o_core, t_core)

    # stack and return
    dice_separate = torch.stack([
        ncr_net_dice, 
        edema_dice, 
        enhancing_dice,
    ], dim=1)            # shape (B,3)

    dice_evaluate = torch.stack([
        dice_whole, 
        dice_core, 
        enhancing_dice, 
        enhancing_dice_postpro,
    ], dim=1)           # shape (B,4)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()





def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def test_softmax(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        mask_name=None):

    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018', 'BRATS2023']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        

    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - 80) / (80 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(80 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 80)

        w_cnt = int(np.ceil((W - 80) / (80 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(80 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 80)

        z_cnt = int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.module.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h+80, w:w+80, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print (msg)
    model.train()
    return vals_evaluation.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





# import numpy as np
# import torch
# import logging
# import time

# def crop(vol):
#     """
#     Determine crop bounds from the non-zero region.
#     Expects vol shape (C, H, W, D) where C is the number of modalities.
#     Returns (x_min, x_max, y_min, y_max, z_min, z_max).
#     """
#     vol_for_crop = np.amax(vol, axis=0)
#     assert len(vol_for_crop.shape) == 3, "Expected 3D volume after collapsing channels."
#     x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol_for_crop != 0)

#     x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
#     y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
#     z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

#     def sup_128(xmin, xmax):
#         if xmax - xmin < 128:
#             ecart = int((128 - (xmax - xmin)) / 2)
#             xmax = xmax + ecart + 1
#             xmin = xmin - ecart
#         if xmin < 0:
#             xmax -= xmin
#             xmin = 0
#         return xmin, xmax

#     x_min, x_max = sup_128(x_min, x_max)
#     y_min, y_max = sup_128(y_min, y_max)
#     z_min, z_max = sup_128(z_min, z_max)

#     return x_min, x_max, y_min, y_max, z_min, z_max

# def sliding_window_inference(model, vol, brain_mask, window_size=80, overlap=0.5):
#     """
#     Performs sliding window inference on a (cropped) volume.
    
#     Inputs:
#       - vol: numpy array with shape [C, H, W, D] (normalized modalities)
#       - brain_mask: numpy array with shape [H, W, D]
#       - window_size: size of the sliding cubic patch.
#       - overlap: fractional overlap between patches.
      
#     Returns:
#       - pred_softmax: softmax probabilities, shape (H, W, D, num_cls)
#       - pred_labels: final label predictions, shape (H, W, D)
#     """
#     # Get spatial dimensions from the cropped volume.
#     _, H, W, D = vol.shape
#     stride = int(window_size * (1 - overlap))
#     device = next(model.parameters()).device

#     # Create containers for accumulation.
#     # (Here we assume num_cls=4; modify if needed for your case)
#     pred = np.zeros((4, H, W, D), dtype=np.float32)
#     count = np.zeros((1, H, W, D), dtype=np.float32)

#     # Prepare sliding window start indices along h, w, d axes.
#     h_steps = list(range(0, H - window_size + 1, stride))
#     if h_steps[-1] != H - window_size:
#         h_steps.append(H - window_size)
#     w_steps = list(range(0, W - window_size + 1, stride))
#     if w_steps[-1] != W - window_size:
#         w_steps.append(W - window_size)
#     d_steps = list(range(0, D - window_size + 1, stride))
#     if d_steps[-1] != D - window_size:
#         d_steps.append(D - window_size)

#     with torch.no_grad():
#         for h in h_steps:
#             for w in w_steps:
#                 for d in d_steps:
#                     h_end = h + window_size
#                     w_end = w + window_size
#                     d_end = d + window_size

#                     # Use the brain mask to zero out irrelevant background.
#                     mask_patch = brain_mask[h:h_end, w:w_end, d:d_end]
#                     # Multiply by the mask (you may adapt if needed).
#                     x_patch = vol[:, h:h_end, w:w_end, d:d_end] * mask_patch

#                     # Prepare tensor input.
#                     x_t = torch.from_numpy(x_patch).unsqueeze(0).to(device).float()
#                     # A dummy modality mask (adjust as needed).
#                     modality_mask = torch.tensor([[True, True, True, True]], device=device)
                    
#                     # Forward pass (assumes model returns logits per class).
#                     output = model(x_t, mask=modality_mask)[0]
#                     output = output.squeeze(0).cpu().numpy()  # shape: [num_cls, window_size, window_size, window_size]

#                     pred[:, h:h_end, w:w_end, d:d_end] += output
#                     count[:, h:h_end, w:w_end, d:d_end] += 1

#     pred = pred / (count + 1e-8)
#     # Compute softmax along the class axis.
#     pred_exp = np.exp(pred)
#     pred_softmax = pred_exp / np.sum(pred_exp, axis=0, keepdims=True)
#     pred_labels = np.argmax(pred_softmax, axis=0)

#     # Transpose softmax to get shape (H, W, D, num_cls)
#     return pred_softmax.transpose(1, 2, 3, 0), pred_labels

# def normalize(vol):
#     """
#     Placeholder normalization function.
#     Modify this function to match your training/inference normalization.
#     For simplicity, here we scale by the maximum value.
#     """
#     vol = vol.astype(np.float32)
#     max_val = np.max(vol)
#     if max_val > 0:
#         vol = vol / max_val
#     return vol

# def softmax_output_dice_class4(output, target):
#     eps = 1e-8
#     # Label 1
#     o1 = (output == 1).float()
#     t1 = (target == 1).float()
#     intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
#     denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
#     ncr_net_dice = intersect1 / denominator1

#     # Label 2
#     o2 = (output == 2).float()
#     t2 = (target == 2).float()
#     intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
#     denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
#     edema_dice = intersect2 / denominator2

#     # Label 3
#     o3 = (output == 3).float()
#     t3 = (target == 3).float()
#     intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
#     denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
#     enhancing_dice = intersect3 / denominator3

#     # Post processing for label 3 (enhancing)
#     if torch.sum(o3) < 500:
#        o4 = o3 * 0.0
#     else:
#        o4 = o3
#     t4 = t3
#     intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
#     denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
#     enhancing_dice_postpro = intersect4 / denominator4

#     # Whole tumor: sum of labels 1, 2, 3.
#     o_whole = o1 + o2 + o3 
#     t_whole = t1 + t2 + t3 
#     intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
#     denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
#     dice_whole = intersect_whole / denominator_whole

#     # Tumor core: labels 1 and 3.
#     o_core = o1 + o3
#     t_core = t1 + t3
#     intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
#     denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
#     dice_core = intersect_core / denominator_core

#     dice_separate = torch.cat((ncr_net_dice.unsqueeze(1), edema_dice.unsqueeze(1), enhancing_dice.unsqueeze(1)), dim=1)
#     dice_evaluate = torch.cat((dice_whole.unsqueeze(1), dice_core.unsqueeze(1), enhancing_dice.unsqueeze(1), enhancing_dice_postpro.unsqueeze(1)), dim=1)

#     return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

# def softmax_output_dice_class5(output, target):
#     eps = 1e-8
#     # Label 1
#     o1 = (output == 1).float()
#     t1 = (target == 1).float()
#     intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
#     denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
#     necrosis_dice = intersect1 / denominator1

#     # Label 2
#     o2 = (output == 2).float()
#     t2 = (target == 2).float()
#     intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
#     denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
#     edema_dice = intersect2 / denominator2

#     # Label 3
#     o3 = (output == 3).float()
#     t3 = (target == 3).float()
#     intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
#     denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
#     non_enhancing_dice = intersect3 / denominator3

#     # Label 4
#     o4 = (output == 4).float()
#     t4 = (target == 4).float()
#     intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
#     denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
#     enhancing_dice = intersect4 / denominator4

#     # Post processing for label 4 (enhancing)
#     if torch.sum(o4) < 500:
#         o5 = o4 * 0
#     else:
#         o5 = o4
#     t5 = t4
#     intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
#     denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
#     enhancing_dice_postpro = intersect5 / denominator5

#     # Whole tumor: labels 1, 2, 3, 4.
#     o_whole = o1 + o2 + o3 + o4
#     t_whole = t1 + t2 + t3 + t4
#     intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
#     denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
#     dice_whole = intersect_whole / denominator_whole

#     # Tumor core: labels 1, 3, 4.
#     o_core = o1 + o3 + o4
#     t_core = t1 + t3 + t4
#     intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
#     denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
#     dice_core = intersect_core / denominator_core

#     dice_separate = torch.cat((necrosis_dice.unsqueeze(1), edema_dice.unsqueeze(1), non_enhancing_dice.unsqueeze(1), enhancing_dice.unsqueeze(1)), dim=1)
#     dice_evaluate = torch.cat((dice_whole.unsqueeze(1), dice_core.unsqueeze(1), enhancing_dice.unsqueeze(1), enhancing_dice_postpro.unsqueeze(1)), dim=1)

#     return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

# class AverageMeter(object):
#     """Computes and stores the average and current value."""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val   = 0
#         self.avg   = 0
#         self.sum   = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val   = val
#         self.sum   += val * n
#         self.count += n
#         self.avg   = self.sum / self.count

# def test_softmax(test_loader, model, dataname='BRATS2020', feature_mask=None, mask_name=None, window_size=80, overlap=0.5):
#     """
#     Offline patch-based evaluation which, for every subject:
#       1. Extracts the multi-modal volume (assumed shape [B, C, H, W, D])
#       2. Constructs a full brain mask from the non-zero voxels
#       3. Crops the volume (and optionally an external mask) using the online crop()
#          function
#       4. Normalizes the cropped volume (using normalize())
#       5. Evaluates the cropped volume with sliding-window inference (using window_size & overlap)
#       6. Reinserts the predicted cropped segmentation into full volume space and computes Dice scores
     
#     Returns:
#       Overall average Dice scores (via the provided metric function).
#     """
#     # For test set assumed full volume dimensions.
#     full_H, full_W, full_T = 240, 240, 155
#     model.eval()
#     vals_evaluation = AverageMeter()
#     vals_separate   = AverageMeter()

#     # Setup dataset parameters.
#     if dataname in ['BRATS2020', 'BRATS2018', 'BRATS2023']:
#         num_cls = 4
#         class_evaluation = ['whole', 'core', 'enhancing', 'enhancing_postpro']
#     elif dataname == 'BRATS2015':
#         num_cls = 5
#         class_evaluation = ['whole', 'core', 'enhancing', 'enhancing_postpro']
#     else:
#         raise ValueError("Unsupported dataset name provided.")

#     for i, data in enumerate(test_loader):
#         # Data order: [x, target, (optional mask), subject_names]
#         # x: [B, C, H, W, D] and target: [B, H, W, D]
#         x     = data[0].cuda()
#         target = data[1].cuda()
#         names  = data[-1]

#         # Process external mask if provided.
#         if feature_mask is not None:
#             mask = torch.from_numpy(np.array(feature_mask))
#             # If feature_mask is provided as [H, W, D], add batch & channel dimensions.
#             if mask.ndim == 3:
#                 mask = mask.unsqueeze(0)  # now shape [1, H, W, D]
#                 mask = mask.unsqueeze(1)  # now shape [1, 1, H, W, D]
#             elif mask.ndim == 4:
#                 # If the feature_mask has shape [B, H, W, D] (batch but no channel),
#                 # then add the channel dimension:
#                 mask = mask.unsqueeze(1)  # now shape [B, 1, H, W, D]
#             # If needed, repeat along the batch dimension to match the number of names.
#             mask = mask.repeat(len(names), 1, 1, 1, 1)
#         else:
#             if len(data) > 2:
#                 mask = data[2]
#             else:
#                 mask = None

#         if mask is not None:
#             mask = mask.cuda()
    

#         # Loop over subjects in the current batch.
#         _, _, H, W, Z = x.size()
#         for b in range(x.size(0)):
#             # Convert full volume to numpy.
#             vol_full = x[b].cpu().numpy()  # shape: [C, H, W, D]
#             full_shape = vol_full.shape[1:]  # (H, W, D)

#             # Create a brain mask from non-zero voxels.
#             brain_mask_full = (vol_full.sum(axis=0) > 0).astype(np.float32)

#             # Determine cropping bounds.
#             x_min, x_max, y_min, y_max, z_min, z_max = crop(vol_full)
#             vol_cropped = vol_full[:, x_min:x_max, y_min:y_max, z_min:z_max]
#             brain_mask_cropped = brain_mask_full[x_min:x_max, y_min:y_max, z_min:z_max]

#             # If an external mask is provided, crop it as well.
#             if mask is not None:
#                 mask_cropped = mask[b:b+1, :, x_min:x_max, y_min:y_max, z_min:z_max]
#             else:
#                 mask_cropped = None

#             # Normalize the cropped volume.
#             # vol_cropped = normalize(vol_cropped)

#             # Run sliding window inference on the cropped volume.
#             pred_softmax, pred_labels_cropped = sliding_window_inference(
#                 model, vol_cropped, brain_mask_cropped,
#                 window_size=window_size, overlap=overlap)

#             # Reconstruct a full-volume segmentation.
#             full_pred = np.zeros(full_shape, dtype=np.uint8)
#             full_pred[x_min:x_max, y_min:y_max, z_min:z_max] = pred_labels_cropped

#             # Convert prediction and target to tensors for Dice computation.
#             full_pred_tensor = torch.from_numpy(full_pred).unsqueeze(0).cuda()  # [1, H, W, D]
#             target_tensor    = target[b].unsqueeze(0)

#             # Compute Dice scores using the appropriate metric.
#             if dataname in ['BRATS2020', 'BRATS2018', 'BRATS2023']:
#                 scores_separate, scores_evaluation = softmax_output_dice_class4(full_pred_tensor, target_tensor)
#             elif dataname == 'BRATS2015':
#                 scores_separate, scores_evaluation = softmax_output_dice_class5(full_pred_tensor, target_tensor)

#             # Print per-subject dice scores.
#             msg = f"Subject: {names[b]}, Dice scores: {dict(zip(class_evaluation, scores_evaluation[0]))}"
#             logging.info(msg)
#             print(msg)

#             vals_evaluation.update(scores_evaluation[0])
#             vals_separate.update(scores_separate[0])

#     avg_msg = 'Average scores: ' + ', '.join([f'{k}: {v:.4f}' for k, v in zip(class_evaluation, vals_evaluation.avg)])
#     print(avg_msg)
#     logging.info(avg_msg)
#     model.train()
#     return vals_evaluation.avg
