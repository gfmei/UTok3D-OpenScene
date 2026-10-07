"""Load OpenScene's LSeg network directly, with every weight supplied by its checkpoint.

Avoid Lightning's training dataset construction and redundant backbone downloads.
The pinned upstream model and evaluation wrapper remain unmodified.
"""
from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'src/lseg_feature_extraction'


def load_model(checkpoint, device='cuda'):
    sys.path.insert(0, str(UPSTREAM))
    import clip
    import timm
    from modules.models.lseg_net import LSegNet
    from metrics import CLASSES

    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {k.removeprefix('net.'): v for k, v in saved['state_dict'].items()
             if k.startswith('net.')}
    clip_state = {k.removeprefix('clip_pretrained.'): v for k, v in state.items()
                  if k.startswith('clip_pretrained.')}
    original_clip_load, original_timm_create = clip.load, timm.create_model

    def checkpoint_clip(*args, **kwargs):
        return clip.model.build_model(dict(clip_state)).cpu(), None

    def uninitialized_vit(*args, **kwargs):
        kwargs['pretrained'] = False
        return original_timm_create(*args, **kwargs)

    clip.load, timm.create_model = checkpoint_clip, uninitialized_vit
    try:
        model = LSegNet(list(CLASSES), backbone='clip_vitl16_384', features=256,
                       arch_option=0, block_depth=0, activation='lrelu')
    finally:
        clip.load, timm.create_model = original_clip_load, original_timm_create
    incompatible = model.load_state_dict(state, strict=True)
    print('Checkpoint loaded strictly:', incompatible, flush=True)
    model.eval().requires_grad_(False)
    # Same q/k/v, scale and projections with PyTorch's fused attention kernel.
    # Validated against the unmodified upstream attention on a real S3DIS image.
    def attention(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads) \
            .permute(2,0,3,1,4).unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale)
        return self.proj_drop(self.proj(x.transpose(1,2).reshape(b,n,c)))
    for block in model.pretrained.model.blocks:
        block.attn.forward = types.MethodType(attention, block.attn)
    return model.to(device)


class Extractor:
    def __init__(self, checkpoint, long_side=320, prompt=True, flip=True):
        import clip
        from metrics import CLASSES
        self.model = load_model(checkpoint)
        # Match OpenScene's single-scale, doubled-base/crop LSeg wrapper.
        from additional_utils.models import LSeg_MultiEvalModule
        self.model.mean = [0.5]*3
        self.model.std = [0.5]*3
        self.model._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.model.evaluate_random = self.model.forward
        self.model.base_size = 2 * long_side
        self.model.crop_size = 2 * long_side
        self.evaluator = LSeg_MultiEvalModule(self.model, scales=[1], flip=flip).cuda().eval()
        self.long_side = long_side
        labels = [f'a {name} in a scene' for name in CLASSES] if prompt else list(CLASSES)
        with torch.inference_mode():
            text = self.model.clip_pretrained.encode_text(clip.tokenize(labels).cuda())
            self.text = (text / text.norm(dim=-1, keepdim=True)).float()

    @torch.inference_mode()
    def __call__(self, rgb):
        import numpy as np
        from PIL import Image
        h, w = rgb.shape[:2]
        scale = self.long_side / max(h, w)
        nh, nw = round(h*scale), round(w*scale)
        resized = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
        x = torch.from_numpy(resized.copy()).permute(2, 0, 1).cuda().float()/255
        x = (x - 0.5) / 0.5
        # Use mixed precision to fit the 20 GB GPU; record this in the run config.
        with torch.autocast('cuda', dtype=torch.float16):
            features = self.evaluator(x[None], '')[0].float()
        if features.shape[1:] != (nh, nw):
            raise ValueError(f'Unexpected feature resolution {features.shape}, expected {nh,nw}')
        return features, (nh, nw)
