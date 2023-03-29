import torch
# ViT-B/16 - ViT-Base
VIT_BASE_PATCH_SIZE=16
VIT_BASE_LAYERS=12
VIT_BASE_HIDDEN_SIZE=768
VIT_BASE_MLP_SIZE=3072
VIT_BASE_HEADS=12
VIT_BASE_ATTN_DROPOUT=0
VIT_BASE_16_DROPOUT=0.1
TRAINING_RESOLUTION=224

# int(TRAINING_RESOLUTION ** 2 / VIT_BASE_PATCH_SIZE)
VIT_BASE_NUM_PATCHES=196

class EmbeddedPatches(torch.nn.Module):
  def __init__(self, in_features):
    super().__init__()

    # patches -> hybrid architecture - feature maps of a CNN
    self.patcher = torch.nn.Conv2d(
      in_channels=in_features,
      out_channels=VIT_BASE_HIDDEN_SIZE,
      kernel_size=VIT_BASE_PATCH_SIZE,
      stride=VIT_BASE_PATCH_SIZE,
      padding=0
    )
    self.flatten_layer = torch.nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x):
    # [[class] embedding; flattened patches] + position embedding
    # [Xclass; x1pE; x2pE; ...; xNpE] + Epos

    # [1, 768, 196]
    x_patched = self.patcher(x)
    x_flattened = self.flatten_layer(x_patched)

    return x_flattened.permute(0, 2, 1)

# MSA(LN(zl-1)) + zl-1
class MSABlock(torch.nn.Module):
  def __init__(self, embed_dim=VIT_BASE_HIDDEN_SIZE, 
                     num_heads=VIT_BASE_HEADS,
                     attn_dropout=VIT_BASE_ATTN_DROPOUT):
    super().__init__()
    self.layer_norm = torch.nn.LayerNorm(normalized_shape=768)
    self.multiheaded_self_attention = torch.nn.MultiheadAttention(
      embed_dim=embed_dim,
      num_heads=num_heads,
      dropout=attn_dropout,
      batch_first=True
    )

  def forward(self, x):
    x = self.layer_norm(x)
    out, _ = self.multiheaded_self_attention(query=x,key=x,value=x, need_weights=False)
    return out

# MLP(LN(zl)) + zl
class MLPHead(torch.nn.Module):
  def __init__(self, droupout=VIT_BASE_16_DROPOUT):
    super().__init__()
    self.layer_norm = torch.nn.LayerNorm(normalized_shape=768)
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(in_features=VIT_BASE_HIDDEN_SIZE, out_features=VIT_BASE_MLP_SIZE),
      torch.nn.GELU(),
      torch.nn.Dropout(p=droupout),
      torch.nn.Linear(in_features=VIT_BASE_MLP_SIZE, out_features=VIT_BASE_HIDDEN_SIZE),
      torch.nn.Dropout(p=droupout),
    )
  def forward(self, x): 
    x = self.layer_norm(x)
    x = self.mlp(x)
    return x

class TransformerEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.msa = MSABlock()
    self.mlp_head = MLPHead()

  def forward(self, x):
    return self.mlp_head(self.msa(x))

class VisionTransformer(torch.nn.Module):
  def __init__(self, in_features, out_features=3):
    super().__init__()
    self.patcher = EmbeddedPatches(in_features)
    self.encoder = TransformerEncoder()
    self.decoder = torch.nn.Sequential(
      torch.nn.LayerNorm(normalized_shape=768),
      torch.nn.Linear(in_features=VIT_BASE_HIDDEN_SIZE, out_features=out_features)
    )
    self.class_embedding = torch.nn.Parameter(data=torch.randn(1, 1, 768), requires_grad=True)
    self.position_embedding = torch.nn.Parameter(torch.randn(1, 196 + 1, 768), requires_grad=True)

  def forward(self, x):
    x = self.patcher(x)
    x = torch.cat((self.class_embedding, x), dim=1)
    x = x + self.position_embedding
    x = self.encoder(x)
    return self.decoder(x)
