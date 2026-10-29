import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride, padding):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape

        x = x.flatten(2) # [batch, filters, H x W] -> ou seja, [B, C, N]
        x = x.transpose(1, 2) # transforma em [B, N, C] (= batches, tokens, dimensões por token)
        x = self.norm(x)

        return x, H, W
    
class EfficientSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio
    ):
        super().__init__()

        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio
            )

            self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):

        B, N, C = x.shape

        # Query
        q = self.q(x)

        # Spatial reduction for K and V
        if self.sr_ratio > 1:

            x_spatial = x.transpose(1, 2)

            x_spatial = x_spatial.reshape(
                B,
                C,
                H,
                W
            )

            x_spatial = self.sr(x_spatial)

            x_reduced = x_spatial.flatten(2)

            x_reduced = x_reduced.transpose(1, 2)

            x_reduced = self.norm(x_reduced)

        else:
            x_reduced = x

        # Key and Value
        k = self.k(x_reduced)
        v = self.v(x_reduced)

        # Multi-head representation
        head_dim = C // self.num_heads

        q = q.reshape(
            B,
            N,
            self.num_heads,
            head_dim
        )

        k = k.reshape(
            B,
            -1,
            self.num_heads,
            head_dim
        )

        v = v.reshape(
            B,
            -1,
            self.num_heads,
            head_dim
        )

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention
        attn = q @ k.transpose(-2, -1)

        attn = attn / (head_dim ** 0.5)

        attn = attn.softmax(dim=-1)

        out = attn @ v

        # Merge heads
        out = out.permute(0, 2, 1, 3)

        out = out.reshape(
            B,
            N,
            C
        )

        # Final projection
        out = self.proj(out)

        return out

class MixFFN(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            dim
        )

    def forward(self, x, H, W):

        B, N, C = x.shape

        x = self.fc1(x)

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            -1,
            H,
            W
        )

        x = self.dwconv(x)

        x = self.act(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        x = self.fc2(x)

        return x

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio,
        mlp_ratio=4
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = MixFFN(
            dim=dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x, H, W):

        x = x + self.attn(
            self.norm1(x),
            H,
            W
        )

        x = x + self.mlp(
            self.norm2(x),
            H,
            W
        )

        return x

class MiTStage(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim,
        num_heads,
        sr_ratio,
        depth,
        kernel_size,
        stride,
        padding,
        mlp_ratio=4
    ):
        super().__init__()

        # Converte a feature map em tokens
        self.patch_embed = OverlapPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Empilha os Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        # Normalização final do estágio
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # Feature map -> tokens
        x, H, W = self.patch_embed(x)

        # Processamento pelos Transformer Blocks
        for block in self.blocks:
            x = block(x, H, W)

        # Normalização
        x = self.norm(x)

        # Tokens -> feature map
        B, N, C = x.shape

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            C,
            H,
            W
        )

        return x

class MiTEncoder(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()

        # Stage 1
        self.stage1 = MiTStage(
            in_channels=in_channels,
            embed_dim=32,
            num_heads=1,
            sr_ratio=8,
            depth=2,
            kernel_size=7,
            stride=4,
            padding=3
        )

        # Stage 2
        self.stage2 = MiTStage(
            in_channels=32,
            embed_dim=64,
            num_heads=2,
            sr_ratio=4,
            depth=2,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Stage 3
        self.stage3 = MiTStage(
            in_channels=64,
            embed_dim=160,
            num_heads=5,
            sr_ratio=2,
            depth=2,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Stage 4
        self.stage4 = MiTStage(
            in_channels=160,
            embed_dim=256,
            num_heads=8,
            sr_ratio=1,
            depth=2,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):

        x1 = self.stage1(x)

        x2 = self.stage2(x1)

        x3 = self.stage3(x2)

        x4 = self.stage4(x3)

        return x1, x2, x3, x4

class MLPDecoder(nn.Module):

    def __init__(
        self,
        in_channels=[32, 64, 160, 256],
        decoder_dim=128,
        num_classes=1
    ):
        super().__init__()

        # Projeções MLP
        self.linear_c1 = nn.Linear(
            in_channels[0], decoder_dim
        )

        self.linear_c2 = nn.Linear(
            in_channels[1], decoder_dim
        )

        self.linear_c3 = nn.Linear(
            in_channels[2], decoder_dim
        )

        self.linear_c4 = nn.Linear(
            in_channels[3], decoder_dim
        )

        # Fusão das quatro escalas
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                decoder_dim * 4,
                decoder_dim,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # Classificador final
        self.classifier = nn.Conv2d(
            decoder_dim,
            num_classes,
            kernel_size=1
        )

    def project_feature(self, x, linear):

        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)

        # Projeção dos canais
        x = linear(x)

        # [B, H*W, decoder_dim] -> [B, decoder_dim, H, W]
        x = x.transpose(1, 2).reshape(
            B, -1, H, W
        )

        return x

    def forward(self, features, output_size=(512, 512)):

        x1, x2, x3, x4 = features

        # Projeta os canais de cada estágio
        x1 = self.project_feature(x1, self.linear_c1)
        x2 = self.project_feature(x2, self.linear_c2)
        x3 = self.project_feature(x3, self.linear_c3)
        x4 = self.project_feature(x4, self.linear_c4)

        # Resolução de referência: Stage 1
        target_size = x1.shape[2:]

        x2 = F.interpolate(
            x2, size=target_size,
            mode="bilinear", align_corners=False
        )

        x3 = F.interpolate(
            x3, size=target_size,
            mode="bilinear", align_corners=False
        )

        x4 = F.interpolate(
            x4, size=target_size,
            mode="bilinear", align_corners=False
        )

        # Concatena as quatro escalas
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # Combina as informações
        x = self.linear_fuse(x)

        # Gera o mapa de logits
        x = self.classifier(x)

        # Redimensiona para a resolução original
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return x


class SegFormer(nn.Module):

    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        # Encoder hierárquico MiT
        self.encoder = MiTEncoder(
            in_channels=in_channels
        )

        # Decoder MLP
        self.decoder = MLPDecoder(
            in_channels=[32, 64, 160, 256],
            decoder_dim=128,
            num_classes=num_classes
        )

    def forward(self, x):

        # Extrai features em quatro escalas
        features = self.encoder(x)

        # Combina as features e gera a segmentação
        output = self.decoder(
            features,
            output_size=x.shape[2:]
        )

        return output