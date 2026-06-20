import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)
    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened
    return mx.reshape(final_embedding_flattened, final_embedding_shape)


def _pack_uint8_weight(weight: mx.array) -> mx.array:
    if weight.dtype != mx.uint8 or weight.shape[-1] % 4 != 0:
        return weight

    shape = (*weight.shape[:-1], weight.shape[-1] // 4, 4)
    weight = weight.reshape(shape).astype(mx.uint32)
    shifts = mx.array([0, 8, 16, 24], dtype=mx.uint32)
    return mx.sum(weight << shifts, axis=-1)


def _sanitize_moe_weights(weights: dict, args):
    num_experts = args.num_local_experts
    pack_shared = (
        args.n_shared_experts == 1
        and args.shared_intermediate_size == args.intermediate_size
    )

    def expert_keys(prefix, name, suffix):
        return [
            f"{prefix}.experts.{expert}.{name}.{suffix}"
            for expert in range(num_experts)
        ]

    def has_all(keys):
        return all(key in weights for key in keys)

    def pop_stack(keys):
        return mx.stack([weights.pop(key) for key in keys])

    def stack_shared(routed, shared, axis):
        """Append a single shared expert (no expert dim) onto a pre-stacked
        [E, ...] routed tensor along the leading expert axis."""
        return mx.concatenate([routed, mx.expand_dims(shared, axis=0)], axis=axis)

    for layer_idx in range(args.num_hidden_layers):
        prefix = f"language_model.model.layers.{layer_idx}.block_sparse_moe"
        # The MLX-converted M3 checkpoints (e.g. mlx-community/MiniMax-M3-*bit)
        # store MoE layers under `mlp.` (not `block_sparse_moe.`) with the routed
        # experts ALREADY stacked into `switch_mlp.{gate,up,down}_proj` and the
        # shared expert kept separate under `shared_experts.*`. Detect and remap
        # that layout to the names/packing this model builds. Dense (non-MoE)
        # layers keep their `mlp.{gate,up,down}_proj` and are left untouched.
        stacked = f"language_model.model.layers.{layer_idx}.mlp"
        if any(
            k.startswith(f"{stacked}.switch_mlp.") for k in weights
        ) or f"{stacked}.gate.weight" in weights:
            # router + routing-bias rename: mlp.gate -> block_sparse_moe.gate
            for tail in ("gate.weight", "gate.scales", "gate.biases",
                         "e_score_correction_bias"):
                src_key = f"{stacked}.{tail}"
                if src_key in weights:
                    weights[f"{prefix}.{tail}"] = weights.pop(src_key)

            for suffix in ("weight", "scales", "biases", "bias"):
                sg = f"{stacked}.switch_mlp.gate_proj.{suffix}"
                su = f"{stacked}.switch_mlp.up_proj.{suffix}"
                sd = f"{stacked}.switch_mlp.down_proj.{suffix}"
                shg = f"{stacked}.shared_experts.gate_proj.{suffix}"
                shu = f"{stacked}.shared_experts.up_proj.{suffix}"
                shd = f"{stacked}.shared_experts.down_proj.{suffix}"
                if pack_shared:
                    if sg in weights and su in weights and shg in weights and shu in weights:
                        gate = weights.pop(sg)
                        up = weights.pop(su)
                        shared_gate = weights.pop(shg)
                        shared_up = weights.pop(shu)
                        # concat gate|up along the output-feature axis (axis=1 for
                        # the [E, out, in] stacked tensors), then append shared.
                        routed_gate_up = mx.concatenate([gate, up], axis=1)
                        shared_gate_up = mx.concatenate([shared_gate, shared_up], axis=0)
                        weights[f"{prefix}.switch_mlp.gate_up_proj.{suffix}"] = (
                            stack_shared(routed_gate_up, shared_gate_up, axis=0)
                        )
                    if sd in weights and shd in weights:
                        down = weights.pop(sd)
                        shared_down = weights.pop(shd)
                        weights[f"{prefix}.switch_mlp.down_proj.{suffix}"] = (
                            stack_shared(down, shared_down, axis=0)
                        )
                else:
                    for src_key, mlx_name in ((sg, "gate_proj"), (su, "up_proj"), (sd, "down_proj")):
                        if src_key in weights:
                            weights[f"{prefix}.switch_mlp.{mlx_name}.{suffix}"] = weights.pop(src_key)
                    for src_key, mlx_name in ((shg, "gate_proj"), (shu, "up_proj"), (shd, "down_proj")):
                        if src_key in weights:
                            weights[f"{prefix}.shared_experts.{mlx_name}.{suffix}"] = weights.pop(src_key)
            continue

        for suffix in ("weight", "scales", "biases", "bias"):
            if pack_shared:
                gate_keys = expert_keys(prefix, "w1", suffix)
                up_keys = expert_keys(prefix, "w3", suffix)
                shared_gate_key = f"{prefix}.shared_experts.gate_proj.{suffix}"
                shared_up_key = f"{prefix}.shared_experts.up_proj.{suffix}"
                if has_all([*gate_keys, *up_keys, shared_gate_key, shared_up_key]):
                    gate = pop_stack(gate_keys)
                    up = pop_stack(up_keys)
                    shared_gate = weights.pop(shared_gate_key)
                    shared_up = weights.pop(shared_up_key)
                    routed_gate_up = mx.concatenate([gate, up], axis=1)
                    shared_gate_up = mx.expand_dims(
                        mx.concatenate([shared_gate, shared_up], axis=0), axis=0
                    )
                    weights[f"{prefix}.switch_mlp.gate_up_proj.{suffix}"] = (
                        mx.concatenate([routed_gate_up, shared_gate_up], axis=0)
                    )

                down_keys = expert_keys(prefix, "w2", suffix)
                shared_down_key = f"{prefix}.shared_experts.down_proj.{suffix}"
                if has_all([*down_keys, shared_down_key]):
                    down = pop_stack(down_keys)
                    shared_down = mx.expand_dims(weights.pop(shared_down_key), axis=0)
                    weights[f"{prefix}.switch_mlp.down_proj.{suffix}"] = (
                        mx.concatenate([down, shared_down], axis=0)
                    )
                continue

            for hf_name, mlx_name in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                keys = expert_keys(prefix, hf_name, suffix)
                if has_all(keys):
                    weights[f"{prefix}.switch_mlp.{mlx_name}.{suffix}"] = pop_stack(
                        keys
                    )


class MiniMaxProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.hidden_act = hidden_act
        self.linear_2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        if self.hidden_act == "silu":
            x = nn.silu(x)
        elif self.hidden_act == "quick_gelu":
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = nn.gelu(x)
        return self.linear_2(x)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy
        num_feature_layers = (
            1
            if isinstance(self.vision_feature_layer, int)
            else len(self.vision_feature_layer)
        )
        projector_input_dim = config.vision_config.hidden_size * num_feature_layers
        self.multi_modal_projector = MiniMaxProjector(
            projector_input_dim,
            config.projector_hidden_size,
            config.text_config.hidden_size,
            config.multimodal_projector_bias,
            config.projector_hidden_act,
        )
        self.patch_merge_mlp = MiniMaxProjector(
            config.text_config.hidden_size * config.vision_config.spatial_merge_size**2,
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            config.patch_merge_bias,
            config.projector_hidden_act,
        )

    def _apply_vision_feature_select_strategy(self, features: mx.array) -> mx.array:
        if self.vision_feature_select_strategy == "full":
            return features
        if self.vision_feature_select_strategy == "default":
            if features.ndim >= 3:
                return features[:, 1:]
            return features[1:]
        raise ValueError(
            "Unexpected feature selection strategy: "
            f"{self.vision_feature_select_strategy}"
        )

    def _select_vision_features(self, hidden_states):
        if isinstance(self.vision_feature_layer, int):
            return self._apply_vision_feature_select_strategy(
                hidden_states[self.vision_feature_layer]
            )

        hs_pool = [
            self._apply_vision_feature_select_strategy(hidden_states[layer_idx])
            for layer_idx in self.vision_feature_layer
        ]
        return mx.concatenate(hs_pool, axis=-1)

    def _compute_visual_features(self, pixel_values: mx.array, grid_thw: mx.array):
        dtype = self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        use_hidden_states = (
            self.vision_feature_layer != -1
            or self.vision_feature_select_strategy != "full"
        )
        if use_hidden_states:
            _, hidden_states = self.vision_tower(
                pixel_values, grid_thw, output_hidden_states=True
            )
            image_features = self._select_vision_features(hidden_states)
        else:
            image_features = self.vision_tower(pixel_values, grid_thw)
        image_features = self.multi_modal_projector(image_features)
        return self._merge_visual_tokens(image_features, grid_thw)

    def encode_image(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        if image_grid_thw is None:
            raise ValueError("MiniMax M3 VL image cache requires image_grid_thw")
        return self._compute_visual_features(pixel_values, image_grid_thw)

    def encode_video(
        self,
        pixel_values: mx.array,
        video_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        if video_grid_thw is None:
            raise ValueError("MiniMax M3 VL video cache requires video_grid_thw")
        return self._compute_visual_features(pixel_values, video_grid_thw)

    def _merge_visual_tokens(self, visual_features: mx.array, grid_thw: mx.array):
        merge_size = self.config.vision_config.spatial_merge_size
        feature_dim = visual_features.shape[-1]
        outputs = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            length = t * h * w
            features = visual_features[offset : offset + length]
            offset += length
            features = features.reshape(
                t,
                h // merge_size,
                w // merge_size,
                merge_size,
                merge_size,
                feature_dim,
            )
            features = features.reshape(-1, merge_size * merge_size * feature_dim)
            outputs.append(self.patch_merge_mlp(features))
        return mx.concatenate(outputs, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)

        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        cached = kwargs.get("cached_image_features", None)
        cached_video = kwargs.get("cached_video_features", None)

        self.language_model._position_ids = None
        self.language_model._rope_deltas = None
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if (
            pixel_values is None
            and pixel_values_videos is None
            and cached is None
            and cached_video is None
        ):
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        image_features = None
        if cached is not None:
            image_features = cached.astype(inputs_embeds.dtype)
        elif pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("MiniMax M3 VL requires image_grid_thw for images")
            image_features = self._compute_visual_features(pixel_values, image_grid_thw)
            image_features = image_features.astype(inputs_embeds.dtype)

        video_features = None
        if cached_video is not None:
            video_features = cached_video.astype(inputs_embeds.dtype)
        elif pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("MiniMax M3 VL requires video_grid_thw for videos")
            video_features = self._compute_visual_features(
                pixel_values_videos, video_grid_thw
            ).astype(inputs_embeds.dtype)

        image_token_id = self.config.image_token_id
        if image_token_id is None:
            image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_id
        if video_token_id is None:
            video_token_id = self.config.video_token_index

        inputs_embeds, visual_mask = self.merge_input_ids_with_visual_features(
            inputs_embeds,
            input_ids,
            image_features=image_features,
            video_features=video_features,
            image_token_index=image_token_id,
            video_token_index=video_token_id,
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_mask,
        )

    @staticmethod
    def merge_input_ids_with_visual_features(
        inputs_embeds,
        input_ids,
        image_features=None,
        video_features=None,
        image_token_index=None,
        video_token_index=None,
    ):
        visual_mask = mx.zeros(input_ids.shape, dtype=mx.bool_)

        def scatter_features(features, token_index, name):
            nonlocal inputs_embeds, visual_mask
            if features is None:
                return

            special_mask = input_ids == token_index
            n_tokens = special_mask.sum()
            special_mask_expanded = mx.broadcast_to(
                special_mask[..., None], inputs_embeds.shape
            )

            n_mask_elements = special_mask_expanded.sum()
            if n_mask_elements != features.size:
                raise ValueError(
                    f"{name} features and {name} tokens do not match: "
                    f"tokens: {n_tokens}, features {features.shape[0]}"
                )

            inputs_embeds = masked_scatter(
                inputs_embeds, special_mask_expanded, features
            )
            visual_mask = visual_mask | special_mask

        scatter_features(image_features, image_token_index, "Image")
        scatter_features(video_features, video_token_index, "Video")
        return inputs_embeds, visual_mask

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_image_mask = (input_ids == image_token_index) | (
            input_ids == video_token_index
        )
        n_image_tokens = special_image_mask.sum()
        special_image_mask = mx.broadcast_to(
            special_image_mask[..., None], inputs_embeds.shape
        )

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds, special_image_mask, image_features
        )
        return inputs_embeds, special_image_mask

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            return self.language_model(
                input_ids,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                **kwargs,
            )

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        kwargs.update(input_embeddings_features.to_dict())
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.", 1)
            elif key.startswith("model.vision_tower."):
                key = key.replace("model.vision_tower.", "vision_tower.", 1)
            elif key.startswith("model.multi_modal_projector."):
                key = key.replace(
                    "model.multi_modal_projector.", "multi_modal_projector.", 1
                )
            elif key.startswith("model.patch_merge_mlp."):
                key = key.replace("model.patch_merge_mlp.", "patch_merge_mlp.", 1)
            # MLX-converted M3-VL checkpoints store the CLIP-style vision tower
            # FLATTENED under `vision_tower.*` (e.g.
            # `vision_tower.embeddings.patch_embedding.weight`,
            # `vision_tower.encoder_layers.N.*`, `vision_tower.pre_layrnorm.*`),
            # dropping the `vision_model` nesting level and using
            # `encoder_layers.N` instead of `encoder.layers.N`. The VL model
            # builds the tower as `vision_tower.vision_model.encoder.layers.N`,
            # so remap the flattened names onto the nested module names.
            if key.startswith("vision_tower.") and not key.startswith(
                "vision_tower.vision_model."
            ):
                key = "vision_tower.vision_model." + key[len("vision_tower."):]
            key = re.sub(
                r"^vision_tower\.vision_model\.encoder_layers\.",
                "vision_tower.vision_model.encoder.layers.",
                key,
            )
            # MLX-converted M3 checkpoints store the Lightning-Indexer
            # projections under a `self_attn.indexer.*` submodule; the language
            # model builds them as flat `self_attn.index_*` attributes. Remap
            # to match (scales/biases ride along so the loader quantizes them).
            key = re.sub(
                r"self_attn\.indexer\.(q|k)_(proj|norm)",
                r"self_attn.index_\1_\2",
                key,
            )
            sanitized_weights[key] = value
        weights.clear()

        scale_keys = {
            key.replace(".weight_scale_inv", ".weight")
            for key in sanitized_weights
            if key.endswith(".weight_scale_inv")
        }
        for weight_key in scale_keys:
            weight = sanitized_weights.get(weight_key)
            if weight is not None:
                sanitized_weights[weight_key] = _pack_uint8_weight(weight)

        for key in list(sanitized_weights):
            if key.endswith(".weight_scale_inv"):
                sanitized_weights[key.replace(".weight_scale_inv", ".scales")] = (
                    sanitized_weights.pop(key)
                )

        # The CLIP-style patch embedding is stored channels-last in the
        # MLX-converted checkpoint as (hidden, T, H, W, C); the vision module
        # declares it (hidden, C, T, H, W). Transpose so the conv weight maps
        # onto the module parameter (load_weights is strict on shape).
        pe_key = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight"
        )
        pe = sanitized_weights.get(pe_key)
        if pe is not None and pe.ndim == 5 and pe.shape[1:] == (
            self.config.vision_config.temporal_patch_size,
            self.config.vision_config.patch_size,
            self.config.vision_config.patch_size,
            self.config.vision_config.num_channels,
        ):
            sanitized_weights[pe_key] = mx.transpose(pe, (0, 4, 1, 2, 3))

        args = self.language_model.args
        _sanitize_moe_weights(sanitized_weights, args)
        return sanitized_weights

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
