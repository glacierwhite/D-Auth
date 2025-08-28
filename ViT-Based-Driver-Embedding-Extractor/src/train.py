import sys
if './' not in sys.path:
	sys.path.append('./')
	
from models.vision_transformer import VisionTransformer

model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )