import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from models.phys_mamba_fusion import PhysMambaFusion


def load_model(checkpoint_path, d_model=128, device='cpu'):
    model = PhysMambaFusion(d_model=d_model).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model, img, dic_seq, device='cpu'):
    """
    img:     (1, 3, H, W)  or (3, H, W) single image
    dic_seq: (1, T, 2, H_d, W_d) or (T, 2, H_d, W_d) DIC sequence
    Returns:
        bbox:     predicted bounding boxes
        risk_mu:  failure risk score
        risk_var: risk uncertainty
        k_mu:     stress intensity factor K_I
        k_var:    K_I uncertainty
        gate_maps: mechanical gating masks
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if dic_seq.dim() == 4:
        dic_seq = dic_seq.unsqueeze(0)

    img = img.to(device)
    dic_seq = dic_seq.to(device)

    with torch.no_grad():
        outputs = model(img, dic_seq)

    risk = outputs['risk_mu'].item()
    k_val = outputs['k_mu'].item()
    risk_unc = outputs['risk_var'].item() ** 0.5
    k_unc = outputs['k_var'].item() ** 0.5

    print(f'Risk Score:        {risk:.4f} ± {risk_unc:.4f}')
    print(f'Stress Factor K_I: {k_val:.4f} ± {k_unc:.4f}')
    print(f'Bbox predictions:  {outputs["bbox"].shape}')

    if k_val > 1.5:
        print('WARNING: High stress intensity detected - potential failure risk!')
    elif risk > 0.7:
        print('WARNING: High failure risk score - manual inspection recommended.')
    else:
        print('STATUS: Component within acceptable integrity bounds.')

    return outputs


def demo():
    device = 'cpu'
    model = PhysMambaFusion(d_model=128).to(device)
    model.eval()

    img = torch.randn(1, 3, 224, 224)
    dic_seq = torch.randn(1, 8, 2, 56, 56)

    print('=== Phys-MambaFusion Inference Demo ===')
    outputs = infer(model, img, dic_seq, device)
    print('Gate map shape:', outputs['gate_maps'][-1].shape)
    print('Strain feat shape:', outputs['strain_feat'].shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.checkpoint is None:
        demo()
    else:
        model = load_model(args.checkpoint, args.d_model, args.device)
        img = torch.randn(1, 3, 224, 224)
        dic_seq = torch.randn(1, 8, 2, 56, 56)
        infer(model, img, dic_seq, args.device)
