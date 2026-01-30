import torch

def get_guidance_params(
    phase,
    noise_pred_obj,
    scale_hand,
    trans_hand,
    rotation_hand,
    device,
    *,
    phase1_hand_lrs,
    phase2_hand_lrs,
    noise_obj_lr1,
    noise_obj_lr2,
    obj_lrs,
    obj_2half_lrs,
    obj_latent_lr=None,
    scale_obj=None,
    trans_obj=None,
    rotation_obj=None,
    ):
    """
    Return optimizer param groups and a phase name for logging.
    """
    def make_param(tensor):
        return tensor.clone().detach().requires_grad_(True)
    
    param_groups = []

    if phase == 1: 
        # --- phase 1: hand-only optimization ---
        scale_hand = make_param(scale_hand)
        trans_hand = make_param(trans_hand)
        rotation_hand = make_param(rotation_hand)
        noise_pred_obj_opt = noise_pred_obj

        param_groups = [
            {'params': [scale_hand], 'lr': phase1_hand_lrs['scale']},
            {'params': [trans_hand], 'lr': phase1_hand_lrs['trans']},
            {'params': [rotation_hand], 'lr': phase1_hand_lrs['rot']},
        ]

    elif phase == 1.5:
        # --- phase 1.5: object transformation optimization ---
        scale_obj = make_param(scale_obj)
        trans_obj = make_param(trans_obj)
        rotation_obj = make_param(rotation_obj)
        noise_pred_obj_opt = make_param(noise_pred_obj)

        param_groups = [
            {'params': [scale_obj], 'lr': obj_2half_lrs['scale']},
            {'params': [trans_obj], 'lr': obj_2half_lrs['trans']},
            {'params': [rotation_obj], 'lr': obj_2half_lrs['rot']},
            {'params': [noise_pred_obj_opt], 'lr': noise_obj_lr1},
        ]

    elif phase == 2:
        # --- phase 2: joint hand + object optimization ---
        # noise_pred_obj = make_param(noise_pred_obj)
        scale_obj = make_param(scale_obj)
        trans_obj = make_param(trans_obj)
        rotation_obj = make_param(rotation_obj)

        scale_hand = make_param(scale_hand)
        trans_hand = make_param(trans_hand)
        rotation_hand = make_param(rotation_hand)

        noise_pred_obj_opt = make_param(noise_pred_obj)
        
        param_groups = [
            {'params': [scale_hand], 'lr': phase2_hand_lrs['scale']},
            {'params': [trans_hand], 'lr': phase2_hand_lrs['trans']},
            {'params': [rotation_hand], 'lr': phase2_hand_lrs['rot']},
            {'params': [scale_obj], 'lr': obj_lrs['scale']},
            {'params': [trans_obj], 'lr': obj_lrs['trans']},
            {'params': [rotation_obj], 'lr': obj_lrs['rot']},
            {'params': [noise_pred_obj_opt], 'lr': noise_obj_lr2},
        ]
    
    else:
        raise ValueError(f"Unknown phase {phase}. Expected 'hand_only (1)', 'obj-only (1.5)' or 'joint_hand_obj (2)'.")

    return param_groups, noise_pred_obj_opt, scale_hand, trans_hand, rotation_hand, scale_obj, trans_obj, rotation_obj