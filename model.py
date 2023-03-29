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

VIT_BASE_NUM_PATCHES=int(TRAINING_RESOLUTION * TRAINING_RESOLUTION // VIT_BASE_PATCH_SIZE ** 2)
#  self.num_patches = (img_size * img_size) // patch_size**2
# VIT_BASE_NUM_PATCHES=196

class EmbeddedPatches(torch.nn.Module):
  def __init__(self, colors, batches):
    super().__init__()

    # patches -> hybrid architecture - feature maps of a CNN
    self.patcher = torch.nn.Conv2d(
      in_channels=colors,
      out_channels=VIT_BASE_HIDDEN_SIZE,
      kernel_size=batches,
      stride=batches,
      padding=0
    )
    self.flatten_layer = torch.nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x):
    # [[class] embedding; flattened patches] + position embedding
    # [Xclass; x1pE; x2pE; ...; xNpE] + Epos
    x_patched = self.patcher(x)
    x_flattened = self.flatten_layer(x_patched)
    x_flattened = x_flattened.permute(0, 2, 1) 
    return x_flattened

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
  def __init__(self, dropout=VIT_BASE_16_DROPOUT):
    super().__init__()
    self.layer_norm = torch.nn.LayerNorm(normalized_shape=768)
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(in_features=VIT_BASE_HIDDEN_SIZE, out_features=VIT_BASE_MLP_SIZE),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=dropout),
      torch.nn.Linear(in_features=VIT_BASE_MLP_SIZE, out_features=VIT_BASE_HIDDEN_SIZE),
      torch.nn.Dropout(p=dropout),
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
    x = self.msa(x) + x       # + x is residual connection
    x = self.mlp_head(x) + x  # + x is residual connection
    return x

class VisionTransformer(torch.nn.Module):
  def __init__(self, batches, out_features=3):
    assert TRAINING_RESOLUTION % VIT_BASE_PATCH_SIZE == 0, f"Image size must be divisible by patch size, image size: {TRAINING_RESOLUTION}, patch size: {VIT_BASE_PATCH_SIZE}."
    super().__init__()
    self.patcher = EmbeddedPatches(colors=3, batches=batches)
    self.encoder = torch.nn.Sequential(
       *[TransformerEncoder() for _ in range(VIT_BASE_LAYERS)]
    )
    self.class_embedding = torch.nn.Parameter(
      torch.randn(batches, 1, VIT_BASE_HIDDEN_SIZE), requires_grad=True)
    self.embedding_dropout = torch.nn.Dropout(p=VIT_BASE_16_DROPOUT)
    self.position_embedding = torch.nn.Parameter(
      torch.randn(batches, VIT_BASE_NUM_PATCHES + 1, VIT_BASE_HIDDEN_SIZE), requires_grad=True)
    self.decoder = torch.nn.Sequential(
      torch.nn.LayerNorm(normalized_shape=VIT_BASE_HIDDEN_SIZE),
      torch.nn.Linear(in_features=VIT_BASE_HIDDEN_SIZE, out_features=out_features)
    )

  def forward(self, x):
    # batch_size = x.shape[0] # 1 or 16
    # print(f'batch_size: {batch_size}')
    # class_token = self.class_embedding.expand(16, -1, -1)
    # to transform [1, 196, 768] to [16, 196, 768]
    # because sometimes we get batch with size 1, for example
    # print(f'Class_token after expand: {class_token.shape}')
#
    x = self.patcher(x)
    # print(f'After patcher: {x.shape}')
    x = torch.cat((self.class_embedding, x), dim=1)
    # print(f'After class_embedding: {x.shape}')
    x = x + self.position_embedding
    # print(f'After position_embedding: {x.shape}')
    x = self.embedding_dropout(x)
    # print(f'After embedding droupout: {x.shape}')
    x = self.encoder(x)
    # print(f'After encoder: {x.shape}')
    x = self.decoder(x[:, 0])
    # print(f'After decoder: {x.shape}')
    return x