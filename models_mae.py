# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # Expand attn_mask to match the shape of (B, num_heads, N, N)
            attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, N)
            attn = attn.masked_fill(attn_mask, float('-inf'))  # Fill masked positions with -inf

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTrans   former backbone
    """

    def __init__(self, args=None, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.args = args
        self.num_patches = self.args.num_patches
        self.similarity_loss = self.args.cos_loss

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches,
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.num_patches ** .5), cls_token=True)
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        if os.environ.get("SKIP_MODEL_INIT", "0") != "1":
    	    self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # Function to initialize parameters to zero
    def initialize_parameters_to_zero(self):
        for param in self.parameters():
            if param.requires_grad:  # Check if the parameter is trainable
                nn.init.constant_(param, 0)  # Set parameter to zero

    def forward_encoder(
        self,
        x,
        attn_mask,
        ids_shuffle=None,
        length=None
    ):
        """
        Encoder for variable-length temporal LanguageBind embeddings.

        x:
            [B, N, C]

        attn_mask:
            [B, N+1]

        ids_shuffle:
            [B, N]

        length:
            [B]
        """

        B, N, C = x.shape

        # --------------------------------------------------
        # Normalize length to shape [B]
        # --------------------------------------------------

        if length is None:
            length = torch.full(
                (B,),
                N,
                dtype=torch.long,
                device=x.device
            )
        elif not torch.is_tensor(length):
            length = torch.tensor(
                length,
                dtype=torch.long,
                device=x.device
            )
        else:
            length = length.to(
                device=x.device,
                dtype=torch.long
            )

        # Scalar -> [1]
        if length.dim() == 0:
            length = length.unsqueeze(0)

        # Make sure shape is [B]
        length = length.reshape(-1)

        if length.numel() == 1 and B > 1:
            length = length.expand(B)

        if length.numel() != B:
            raise ValueError(
                "length must contain one value per batch element. "
                f"Got length shape {tuple(length.shape)} for batch size {B}."
            )

        # --------------------------------------------------
        # Normalize ids_shuffle
        # --------------------------------------------------

        if ids_shuffle is not None:

            if not torch.is_tensor(ids_shuffle):
                ids_shuffle = torch.tensor(
                    ids_shuffle,
                    dtype=torch.long,
                    device=x.device
                )
            else:
                ids_shuffle = ids_shuffle.to(
                    device=x.device,
                    dtype=torch.long
                )

            if ids_shuffle.dim() == 1:
                ids_shuffle = ids_shuffle.unsqueeze(0)

        # --------------------------------------------------
        # Positional embeddings
        # --------------------------------------------------

        pos_embed_shuffled_batch = []

        if ids_shuffle is not None:

            for i in range(B):

                # Convert the individual length to Python int.
                # IMPORTANT: do not use length directly as a
                # tensor slice index.
                L = int(length[i].item())

                # Keep L inside valid range.
                L = max(
                    0,
                    min(
                        L,
                        self.num_patches,
                        N
                    )
                )

                # --------------------------------------------------
                # Get positional embeddings for actual sequence
                # --------------------------------------------------

                pos_embed = self.pos_embed[
                    :,
                    1:L + 1,
                    :
                ]

                # --------------------------------------------------
                # Get shuffled indices
                # --------------------------------------------------

                ids = ids_shuffle[i][:L]

                # Shape:
                # [L] -> [1, L, 1]
                ids = ids.reshape(
                    1,
                    L,
                    1
                )

                # Expand to embedding dimension
                ids = ids.expand(
                    1,
                    L,
                    self.pos_embed.shape[-1]
                )

                # --------------------------------------------------
                # Gather shuffled positional embeddings
                # --------------------------------------------------

                pos_embed_shuffled = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids
                )

                # --------------------------------------------------
                # Pad to N
                # --------------------------------------------------

                padding_length = N - L

                if padding_length > 0:

                    padding = torch.zeros(
                        1,
                        padding_length,
                        C,
                        device=x.device,
                        dtype=x.dtype
                    )

                    pos_embed_shuffled = torch.cat(
                        (
                            pos_embed_shuffled,
                            padding
                        ),
                        dim=1
                    )

                elif padding_length < 0:

                    pos_embed_shuffled = (
                        pos_embed_shuffled[:, :N, :]
                    )

                pos_embed_shuffled_batch.append(
                    pos_embed_shuffled
                )

            # --------------------------------------------------
            # Stack batch
            # --------------------------------------------------

            pos_embed_shuffled_batch = torch.cat(
                pos_embed_shuffled_batch,
                dim=0
            )

            # Add positional embeddings
            x = x + pos_embed_shuffled_batch

        else:

            x = x + self.pos_embed[
                :,
                1:N + 1,
                :
            ]

        # --------------------------------------------------
        # CLS token
        # --------------------------------------------------

        cls_token = (
            self.cls_token +
            self.pos_embed[:, :1, :]
        )

        cls_tokens = cls_token.expand(
            B,
            -1,
            -1
        )

        x = torch.cat(
            (
                cls_tokens,
                x
            ),
            dim=1
        )

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        # attn_mask has shape [B, N+1].
        # This matches x after adding CLS.
        for blk in self.blocks:

            x = blk(
                x,
                attn_mask
            )

        x = self.norm(x)

        return x

    def forward_decoder(
        self,
        x,
        ids_restore,
        masked_length,
        length,
        mask
    ):
        """
        Decoder for variable-length temporal embeddings.

        x:
            Encoder output [B, N+1, embed_dim]

        ids_restore:
            [B, N]

        masked_length:
            [B]

        length:
            [B]

        mask:
            [B, N]
        """

        # --------------------------------------------------
        # Normalize length
        # --------------------------------------------------

        B = x.shape[0]

        if not torch.is_tensor(length):
            length = torch.tensor(
                length,
                dtype=torch.long,
                device=x.device
            )
        else:
            length = length.to(
                device=x.device,
                dtype=torch.long
            )

        if length.dim() == 0:
            length = length.unsqueeze(0)

        length = length.reshape(-1)

        if length.numel() == 1 and B > 1:
            length = length.expand(B)

        # Normalize masked_length
        # --------------------------------------------------

        if not torch.is_tensor(masked_length):
            masked_length = torch.tensor(
                masked_length,
                dtype=torch.long,
                device=x.device
            )
        else:
            masked_length = masked_length.to(
                device=x.device,
                dtype=torch.long
            )

        if masked_length.dim() == 0:
            masked_length = masked_length.unsqueeze(0)

        masked_length = masked_length.reshape(-1)

        if masked_length.numel() == 1 and B > 1:
            masked_length = masked_length.expand(B)

        # --------------------------------------------------
        # Normalize ids_restore
        # --------------------------------------------------

        if not torch.is_tensor(ids_restore):
            ids_restore = torch.tensor(
                ids_restore,
                dtype=torch.long,
                device=x.device
            )
        else:
            ids_restore = ids_restore.to(
                device=x.device,
                dtype=torch.long
            )

        if ids_restore.dim() == 1:
            ids_restore = ids_restore.unsqueeze(0)

        # --------------------------------------------------
        # Decoder embedding
        # --------------------------------------------------

        x = self.decoder_embed(x)

        B, N, C = x.shape

        decoded_sequences = []
        attn_masks = []

        # --------------------------------------------------
        # Process every sample
        # --------------------------------------------------

        for i in range(B):

            # Convert scalar tensor to Python integer
            L_full = int(length[i].item())

            L_masked = int(
                masked_length[i].item()
            )

            # Keep values valid
            L_full = max(
                0,
                min(
                    L_full,
                    self.num_patches,
                    N - 1
                )
            )

            L_masked = max(
                0,
                min(
                    L_masked,
                    L_full
                )
            )

            # --------------------------------------------------
            # Number of mask tokens
            # --------------------------------------------------

            mask_tokens_needed = L_full - L_masked

            mask_tokens = self.mask_token.repeat(
                1,
                mask_tokens_needed,
                1
            )

            # --------------------------------------------------
            # Remove CLS token
            # Keep only encoded unmasked tokens
            # --------------------------------------------------

            x_seq = x[
                i,
                1:L_masked + 1,
                :
            ].unsqueeze(0)

            # --------------------------------------------------
            # Add mask tokens
            # --------------------------------------------------

            x_full = torch.cat(
                [
                    x_seq,
                    mask_tokens
                ],
                dim=1
            )

            # --------------------------------------------------
            # Restore original temporal ordering
            # --------------------------------------------------

            restore_ids = ids_restore[
                i,
                :L_full
            ]

            restore_ids = restore_ids.reshape(
                1,
                L_full,
                1
            )

            restore_ids = restore_ids.expand(
                1,
                L_full,
                C
            )

            x_restored = torch.gather(
                x_full,
                dim=1,
                index=restore_ids
            )

            # --------------------------------------------------
            # Add CLS token
            # --------------------------------------------------

            cls_token = x[
                i,
                :1,
                :
            ].unsqueeze(0)

            x_restored = torch.cat(
                [
                    cls_token,
                    x_restored
                ],
                dim=1
            )

            # --------------------------------------------------
            # Decoder positional embeddings
            # --------------------------------------------------

            x_restored_pos_embed = (
                x_restored
                +
                self.decoder_pos_embed[
                    :,
                    :L_full + 1,
                    :
                ]
            )

            # --------------------------------------------------
            # Pad sequence to N
            # --------------------------------------------------

            padding_length = (
                N - (L_full + 1)
            )

            if padding_length > 0:

                padding = torch.zeros(
                    1,
                    padding_length,
                    C,
                    device=x.device,
                    dtype=x.dtype
                )

                x_restored_pos_embed = torch.cat(
                    [
                        x_restored_pos_embed,
                        padding
                    ],
                    dim=1
                )

            elif padding_length < 0:

                x_restored_pos_embed = (
                    x_restored_pos_embed[:, :N, :]
                )

                padding_length = 0

            # --------------------------------------------------
            # Attention mask
            #
            # False = valid token
            # True  = padding token
            # --------------------------------------------------

            attn_mask = torch.cat(
                [
                    torch.zeros(
                        L_full + 1,
                        dtype=torch.bool,
                        device=x.device
                    ),
                    torch.ones(
                        padding_length,
                        dtype=torch.bool,
                        device=x.device
                    )
                ],
                dim=0
            ).unsqueeze(0)

            decoded_sequences.append(
                x_restored_pos_embed
            )

            attn_masks.append(
                attn_mask
            )

        # --------------------------------------------------
        # Stack batch
        # --------------------------------------------------

        x = torch.cat(
            decoded_sequences,
            dim=0
        )

        attn_mask = torch.cat(
            attn_masks,
            dim=0
        )

        # --------------------------------------------------
        # Decoder Transformer blocks
        # --------------------------------------------------

        for blk in self.decoder_blocks:

            x = blk(
                x,
                attn_mask
            )

        x = self.decoder_norm(x)

        # --------------------------------------------------
        # Predict original 1024-dimensional embeddings
        # --------------------------------------------------

        x = self.decoder_pred(x)

        # Remove CLS token
        x = x[:, 1:, :]

        return x
    def forward_loss(self, imgs, pred, mask, length, temperature=0.07):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs.clone()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        if self.similarity_loss:
            loss = 1 - self.cos_loss(pred, target)
        else:
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, masked_padded_embeddings, ids_restore, ids_shuffle, masked_length, normalized_stacked_embeddings,
                length, mask, attn_mask=None, mask_ratio=0.75):
        latent = self.forward_encoder(masked_padded_embeddings, attn_mask, ids_shuffle=ids_shuffle, length=length)
        pred = self.forward_decoder(latent, ids_restore, masked_length, length, mask)  # [N, L, p*p*3]
        loss = self.forward_loss(normalized_stacked_embeddings, pred, mask, length)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch192_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=kwargs['args'].num_patches, embed_dim=kwargs['args'].input_size, depth=kwargs['args'].depth, num_heads=kwargs['args'].num_heads,
        decoder_embed_dim=kwargs['args'].decoder_embed_dim, decoder_depth=kwargs['args'].decoder_depth, decoder_num_heads=kwargs['args'].decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_custom = mae_vit_small_patch192_dec512d8b  # decoder: 512 dim, 8 blocks
