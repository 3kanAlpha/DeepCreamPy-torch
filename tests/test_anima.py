import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

import anima
import config
import model
from anima import (
  ANIMA_DEFAULT_NEGATIVE_PROMPT,
  ANIMA_SUBDIRS,
  AnimaCancelledError,
  AnimaInpainter,
  AnimaSettings,
  _build_soft_latent_mask,
  _convert_anima_diffusion_state_dict,
  _convert_qwen3_state_dict,
  _ensure_cosmos_torchvision_shim,
  _fill_masked_pixels_for_vae,
  _load_state_dict_with_dtype,
  anima_availability,
  anima_default_fp8_storage,
  anima_dependencies_available,
  anima_tokenizers_dir,
)
from anima_lllite import (
  ControlNetLLLiteDiT,
  LLLiteConfig,
  LLLiteModuleDiT,
  _Conditioning1,
  build_inpaint_cond_image,
  infer_lllite_config,
  map_lllite_name_to_diffusers,
)
from decensor import Decensor, ProcessingCancelledError
from libs.utils import (
  group_anima_regions,
  prepare_anima_crop,
  resize_anima_prediction,
)
from main import MainWindow
from mask_detect import apply_mask_to_image
from model import create_inpainter


class AnimaLLLiteTests(unittest.TestCase):
  def test_lllite_name_mapping_covers_comfyui_and_diffusers_targets(self):
    self.assertEqual(
      map_lllite_name_to_diffusers("lllite_dit_blocks_0_self_attn_q_proj"),
      "transformer_blocks.0.attn1.to_q",
    )
    self.assertEqual(
      map_lllite_name_to_diffusers("lllite_dit_blocks_12_self_attn_k_proj"),
      "transformer_blocks.12.attn1.to_k",
    )
    self.assertEqual(
      map_lllite_name_to_diffusers("lllite_dit_blocks_5_cross_attn_q_proj"),
      "transformer_blocks.5.attn2.to_q",
    )
    self.assertEqual(
      map_lllite_name_to_diffusers("lllite_dit_blocks_27_mlp_layer1"),
      "transformer_blocks.27.ff.net.0.proj",
    )
    self.assertEqual(
      map_lllite_name_to_diffusers("blocks.3.self_attn.output_proj"),
      "transformer_blocks.3.attn1.to_out.0",
    )
    self.assertIsNone(map_lllite_name_to_diffusers("llm_adapter_blocks_0_q_proj"))

  def test_build_inpaint_cond_image_masks_rgb_and_scales_mask(self):
    rgb = torch.full((1, 3, 16, 16), 0.75, dtype=torch.float32)
    mask = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    mask[:, :, 4:12, 4:12] = 1.0

    cond = build_inpaint_cond_image(rgb, mask, masked_input=True)
    self.assertEqual(tuple(cond.shape), (1, 4, 16, 16))
    # Inside mask: RGB zeroed, mask channel = +1.0
    self.assertTrue(torch.equal(cond[:, :3, 4:12, 4:12], torch.zeros(1, 3, 8, 8)))
    self.assertTrue(torch.equal(cond[:, 3:, 4:12, 4:12], torch.ones(1, 1, 8, 8)))
    # Outside mask: RGB preserved, mask channel = -1.0
    self.assertTrue(
      torch.allclose(cond[:, :3, 0:4, 0:4], torch.full((1, 3, 4, 4), 0.75))
    )
    self.assertTrue(torch.equal(cond[:, 3:, 0:4, 0:4], torch.full((1, 1, 4, 4), -1.0)))

  @unittest.skipUnless(
    anima_dependencies_available(), "Optional anima dependencies not installed"
  )
  def test_lllite_safetensors_hooks_tiny_cosmos_transformer(self):
    from diffusers.models import CosmosTransformer3DModel
    from safetensors.torch import save_file

    _ensure_cosmos_torchvision_shim()
    torch.manual_seed(0)
    transformer = CosmosTransformer3DModel(
      in_channels=4,
      out_channels=4,
      num_attention_heads=2,
      attention_head_dim=16,
      num_layers=2,
      mlp_ratio=2.0,
      text_embed_dim=32,
      adaln_lora_dim=16,
      max_size=(4, 32, 32),
      patch_size=(1, 2, 2),
      rope_scale=(1.0, 4.0, 4.0),
      concat_padding_mask=True,
      extra_pos_embed_type=None,
    ).eval()

    # Build a synthetic v2 LLLite checkpoint with ResBlock + ASPP + 2 hooked linears.
    cond_trunk = _Conditioning1(
      cond_dim=16,
      cond_emb_dim=8,
      n_resblocks=1,
      use_aspp=True,
      aspp_dilations=(1, 2),
      cond_in_channels=4,
    )
    mod0 = LLLiteModuleDiT(
      "lllite_dit_blocks_0_self_attn_q_proj",
      transformer.transformer_blocks[0].attn1.to_q,
      cond_emb_dim=8,
      mlp_dim=12,
    )
    mod1 = LLLiteModuleDiT(
      "lllite_dit_blocks_1_mlp_layer1",
      transformer.transformer_blocks[1].ff.net[0].proj,
      cond_emb_dim=8,
      mlp_dim=12,
    )
    # Give up/film non-zero weights so injection visibly perturbs transformer output.
    for mod in (mod0, mod1):
      torch.nn.init.normal_(mod.up.weight, std=0.2)
      torch.nn.init.normal_(mod.cond_to_film.weight, std=0.2)
      torch.nn.init.normal_(mod.depth_embed, std=0.1)

    state_dict = {
      f"lllite_conditioning1.{k}": v.detach().cpu()
      for k, v in cond_trunk.state_dict().items()
    }
    for mod in (mod0, mod1):
      for k, v in mod.state_dict().items():
        state_dict[f"{mod.lllite_name}.{k}"] = v.detach().cpu()

    with tempfile.TemporaryDirectory() as tmp:
      ckpt_path = Path(tmp) / "test_inpaint_lllite.safetensors"
      save_file(
        state_dict,
        str(ckpt_path),
        metadata={
          "lllite.cond_in_channels": "4",
          "lllite.cond_dim": "16",
          "lllite.cond_emb_dim": "8",
          "lllite.cond_resblocks": "1",
          "lllite.use_aspp": "true",
          "lllite.aspp_dilations": "1,2",
          "lllite.inpaint_masked_input": "true",
        },
      )

      # Also confirm config inference works without metadata.
      inferred = infer_lllite_config(state_dict, metadata={})
      self.assertEqual(
        inferred,
        LLLiteConfig(
          cond_in_channels=4,
          cond_dim=16,
          cond_emb_dim=8,
          n_resblocks=1,
          use_aspp=True,
          aspp_dilations=(1, 2),
          inpaint_masked_input=True,
        ),
      )

      lllite = ControlNetLLLiteDiT.from_safetensors(
        ckpt_path, transformer, multiplier=1.0, dtype=torch.float32
      )
      self.assertEqual(len(lllite.lllite_modules), 2)

      # Image 64x64 -> latent 8x8 -> patch (1,2,2) -> 4x4 = 16 tokens; stride 16 matches!
      latents = torch.randn(1, 4, 1, 8, 8)
      timestep = torch.tensor([0.5])
      enc = torch.randn(1, 10, 32)
      pad_mask = torch.zeros(1, 1, 64, 64)

      with torch.inference_mode():
        baseline = transformer(
          hidden_states=latents,
          timestep=timestep,
          encoder_hidden_states=enc,
          padding_mask=pad_mask,
          return_dict=False,
        )[0]

        rgb = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        cond_emb = lllite.encode_conditioning(
          rgb,
          mask,
          device=torch.device("cpu"),
          dtype=torch.float32,
          offload_trunk_to_cpu=False,
        )
        self.assertEqual(tuple(cond_emb.shape), (1, 16, 8))

        lllite.apply_to()
        guided = transformer(
          hidden_states=latents,
          timestep=timestep,
          encoder_hidden_states=enc,
          padding_mask=pad_mask,
          return_dict=False,
        )[0]
        lllite.restore()
        restored = transformer(
          hidden_states=latents,
          timestep=timestep,
          encoder_hidden_states=enc,
          padding_mask=pad_mask,
          return_dict=False,
        )[0]

      self.assertTrue(torch.isfinite(guided).all().item())
      self.assertFalse(torch.allclose(baseline, guided))
      self.assertTrue(torch.equal(baseline, restored))

  def test_lllite_rejects_non_4channel_checkpoint(self):
    dummy_sd = {
      "lllite_conditioning1.conv1.weight": torch.zeros(8, 3, 4, 4),
      "lllite_conditioning1.conv3.weight": torch.zeros(16, 8, 4, 4),
      "lllite_conditioning1.proj.weight": torch.zeros(8, 16, 1, 1),
    }
    with self.assertRaisesRegex(ValueError, "4-channel"):
      ControlNetLLLiteDiT.from_state_dict(dummy_sd, torch.nn.Linear(4, 4))


class AnimaPreprocessingAndConversionTests(unittest.TestCase):
  def test_group_anima_regions_whole_image_and_clustering(self):
    # Medium image <= 1024x1024 -> single whole-image pass
    r1 = {(100, 100), (101, 100)}
    r2 = {(700, 700), (701, 700)}
    grouped_small = group_anima_regions((900, 900), [r1, r2], max_size=1024)
    self.assertEqual(len(grouped_small), 1)
    self.assertEqual(grouped_small[0][0], (0, 0, 900, 900))
    self.assertEqual(grouped_small[0][1], r1 | r2)

    # Large image with 2 nearby regions -> merged into 1 crop
    near_a = {(1000, 1000), (1040, 1010)}
    near_b = {(1120, 1020), (1160, 1030)}
    grouped_near = group_anima_regions((2400, 2400), [near_a, near_b], max_size=1024)
    self.assertEqual(len(grouped_near), 1)
    box = grouped_near[0][0]
    self.assertEqual(grouped_near[0][1], near_a | near_b)
    self.assertTrue(
      all(box[0] <= x < box[2] and box[1] <= y < box[3] for x, y in near_a | near_b)
    )

    # Large image with 2 far-apart regions -> 2 separate crops
    far_a = {(100, 100), (140, 110)}
    far_b = {(2200, 2200), (2240, 2210)}
    grouped_far = group_anima_regions((2400, 2400), [far_a, far_b], max_size=1024)
    self.assertEqual(len(grouped_far), 2)

  def test_prepare_anima_crop_alignment_and_no_green_bleed(self):
    # Downscale large crop: dimensions divisible by 16 and thin mask survives
    img = np.full((1500, 1200, 3), (200, 175, 160), dtype=np.uint8)
    marked = np.zeros((1500, 1200), dtype=bool)
    marked[500:800, 600] = True
    img[marked] = (0, 255, 0)
    rgb, missing, unpadded = prepare_anima_crop(
      Image.fromarray(img), marked, (0, 0, 1200, 1500), max_size=1024
    )
    self.assertEqual(rgb.shape[0] % 16, 0)
    self.assertEqual(rgb.shape[1] % 16, 0)
    self.assertLessEqual(rgb.shape[0], 1024)
    self.assertTrue(missing[unpadded].any())
    # Resize prediction roundtrip preserves target size
    pred_f = resize_anima_prediction(
      np.full(
        (rgb[unpadded].shape[0], rgb[unpadded].shape[1], 3), 0.5, dtype=np.float32
      ),
      (1200, 1500),
    )
    self.assertEqual(pred_f.shape, (1500, 1200, 3))

    # Upscale small crop without bleeding green into known pixels
    small = np.full((160, 200, 3), (120, 120, 120), dtype=np.uint8)
    small_mask = np.zeros((160, 200), dtype=bool)
    small_mask[60:90, 80:120] = True
    small[small_mask] = (0, 255, 0)
    rgb_up, miss_up, unpad_up = prepare_anima_crop(
      Image.fromarray(small), small_mask, (0, 0, 200, 160), min_size=512, max_size=1024
    )
    self.assertEqual(rgb_up.shape[0] % 16, 0)
    self.assertEqual(rgb_up.shape[1] % 16, 0)
    self.assertGreaterEqual(max(rgb_up[unpad_up].shape[:2]), 512)
    valid = rgb_up[~miss_up]
    self.assertLess(
      int(np.max(np.abs(valid[:, 1].astype(int) - valid[:, 0].astype(int)))), 2
    )

  def test_fill_masked_pixels_for_vae_removes_green_and_soft_latent_mask(self):
    rgb = np.full((64, 64, 3), (180, 140, 120), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True
    rgb[mask] = (0, 255, 0)

    filled = _fill_masked_pixels_for_vae(rgb, mask)
    np.testing.assert_array_equal(filled[~mask], rgb[~mask])
    np.testing.assert_allclose(
      filled[mask], np.broadcast_to((180, 140, 120), filled[mask].shape), atol=1
    )

    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
    soft = _build_soft_latent_mask(mask_t, 8, 8)
    self.assertEqual(tuple(soft.shape), (1, 1, 1, 8, 8))
    self.assertEqual(float(soft[0, 0, 0, 3, 3]), 1.0)
    self.assertEqual(float(soft[0, 0, 0, 2, 2]), 0.5)
    self.assertEqual(float(soft[0, 0, 0, 0, 0]), 0.0)

  @unittest.skipUnless(
    anima_dependencies_available(), "Optional anima dependencies not installed"
  )
  def test_state_dict_conversion_and_offline_tokenizers(self):
    raw_diff = {
      "diffusion_model.llm_adapter.embed.weight": torch.zeros(4, 4),
      "diffusion_model.blocks.0.self_attn.q_proj.weight": torch.zeros(4, 4),
      "diffusion_model.x_embedder.proj.1.weight": torch.zeros(4, 4),
    }
    tr_sd, cond_sd = _convert_anima_diffusion_state_dict(raw_diff)
    self.assertIn("embed.weight", cond_sd)
    self.assertIn("transformer_blocks.0.attn1.to_q.weight", tr_sd)
    self.assertIn("patch_embed.proj.weight", tr_sd)

    qwen_sd = _convert_qwen3_state_dict(
      {
        "model.embed_tokens.weight": torch.zeros(4, 4),
        "lm_head.weight": torch.zeros(4, 4),
      }
    )
    self.assertIn("embed_tokens.weight", qwen_sd)
    self.assertNotIn("lm_head.weight", qwen_sd)

    # Verify bundled offline tokenizers load strictly offline.
    from transformers import AutoTokenizer, T5TokenizerFast

    tok_dir = anima_tokenizers_dir()
    q_tok = AutoTokenizer.from_pretrained(str(tok_dir / "qwen3"), local_files_only=True)
    t_tok = T5TokenizerFast.from_pretrained(str(tok_dir / "t5"), local_files_only=True)
    self.assertGreater(len(q_tok("1girl, @artist").input_ids), 0)
    self.assertGreater(len(t_tok("1girl, @artist").input_ids), 0)

  @unittest.skipUnless(
    anima_dependencies_available(), "Optional anima dependencies not installed"
  )
  def test_load_state_dict_with_dtype_avoids_diffusers_cast_warnings(self):
    from diffusers.models import AnimaTextConditioner, CosmosTransformer3DModel

    tc = AnimaTextConditioner(
      source_dim=16,
      target_dim=16,
      model_dim=16,
      num_layers=1,
      num_attention_heads=2,
      target_vocab_size=32,
    ).eval()
    tr = CosmosTransformer3DModel(
      in_channels=4,
      out_channels=4,
      num_attention_heads=2,
      attention_head_dim=16,
      num_layers=1,
      mlp_ratio=2.0,
      text_embed_dim=16,
      adaln_lora_dim=16,
      max_size=(4, 16, 16),
      patch_size=(1, 2, 2),
      extra_pos_embed_type="learnable",
    ).eval()

    with self.assertNoLogs("diffusers", level="WARNING"):
      _load_state_dict_with_dtype(tc, tc.state_dict(), torch.bfloat16)
      _load_state_dict_with_dtype(tr, tr.state_dict(), torch.bfloat16)

    self.assertEqual(tc.embed.weight.dtype, torch.bfloat16)
    self.assertEqual(tr.proj_out.weight.dtype, torch.bfloat16)
    # _keep_in_fp32_modules ('learnable_pos_embed') stays in float32.
    self.assertEqual(tr.learnable_pos_embed.pos_emb_h.dtype, torch.float32)

  def test_fp8_storage_defaults_for_8gb_vs_16gb_and_cli_flags(self):
    with patch.object(anima, "anima_gpu_vram_gib", return_value=8.0):
      self.assertTrue(anima_default_fp8_storage())
      self.assertTrue(AnimaSettings().normalized().use_fp8_storage)
      self.assertFalse(
        AnimaSettings(use_fp8_storage=False).normalized().use_fp8_storage
      )

    with patch.object(anima, "anima_gpu_vram_gib", return_value=16.0):
      self.assertFalse(anima_default_fp8_storage())
      self.assertFalse(AnimaSettings().normalized().use_fp8_storage)
      self.assertTrue(AnimaSettings(use_fp8_storage=True).normalized().use_fp8_storage)

    with patch.object(anima, "anima_gpu_vram_gib", return_value=None):
      self.assertFalse(anima_default_fp8_storage())

    with patch.object(sys, "argv", [sys.argv[0]]):
      self.assertIsNone(config.get_args(cli=True).anima_fp8)
    with patch.object(sys, "argv", [sys.argv[0], "--anima-fp8"]):
      self.assertTrue(config.get_args(cli=True).anima_fp8)
    with patch.object(sys, "argv", [sys.argv[0], "--no-anima-fp8"]):
      self.assertFalse(config.get_args(cli=True).anima_fp8)

    with patch.object(sys, "argv", [sys.argv[0]]):
      decensor = Decensor()
    decensor.model_name = "anima"
    decensor.anima_settings = AnimaSettings(use_fp8_storage=True)
    key_fp8 = decensor._current_model_config_key()
    decensor.anima_settings = AnimaSettings(use_fp8_storage=False)
    key_bf16 = decensor._current_model_config_key()
    self.assertNotEqual(key_fp8, key_bf16)

  @unittest.skipUnless(
    anima_dependencies_available(), "Optional anima dependencies not installed"
  )
  def test_fp8_layerwise_casting_compatible_with_lllite_hooks(self):
    from diffusers.models import CosmosTransformer3DModel

    _ensure_cosmos_torchvision_shim()
    tr = CosmosTransformer3DModel(
      in_channels=4,
      out_channels=4,
      num_attention_heads=2,
      attention_head_dim=16,
      num_layers=1,
      mlp_ratio=2.0,
      text_embed_dim=32,
      adaln_lora_dim=16,
      max_size=(4, 32, 32),
      patch_size=(1, 2, 2),
      rope_scale=(1.0, 4.0, 4.0),
      concat_padding_mask=True,
      extra_pos_embed_type=None,
    ).eval()
    _load_state_dict_with_dtype(tr, tr.state_dict(), torch.bfloat16)
    tr.enable_layerwise_casting(
      storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
    )
    self.assertEqual(
      tr.transformer_blocks[0].attn1.to_q.weight.dtype, torch.float8_e4m3fn
    )

    cond_trunk = _Conditioning1(16, 8, 1, False, cond_in_channels=4)
    mod0 = LLLiteModuleDiT(
      "lllite_dit_blocks_0_self_attn_q_proj",
      tr.transformer_blocks[0].attn1.to_q,
      8,
      12,
    )
    torch.nn.init.normal_(mod0.up.weight, std=0.2)
    sd = {f"lllite_conditioning1.{k}": v for k, v in cond_trunk.state_dict().items()}
    sd.update({f"{mod0.lllite_name}.{k}": v for k, v in mod0.state_dict().items()})
    lllite = ControlNetLLLiteDiT.from_state_dict(sd, tr, dtype=torch.bfloat16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr.to(device)
    latents = torch.randn(1, 4, 1, 8, 8, device=device, dtype=torch.bfloat16)
    t = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
    enc = torch.randn(1, 8, 32, device=device, dtype=torch.bfloat16)
    pad = torch.zeros(1, 1, 64, 64, device=device, dtype=torch.bfloat16)
    rgb = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    with torch.inference_mode():
      base = tr(latents, t, enc, padding_mask=pad, return_dict=False)[0]
      lllite.encode_conditioning(
        rgb, mask, device=device, dtype=torch.bfloat16, offload_trunk_to_cpu=False
      )
      lllite.apply_to()
      guided = tr(latents, t, enc, padding_mask=pad, return_dict=False)[0]
      lllite.restore()
      restored = tr(latents, t, enc, padding_mask=pad, return_dict=False)[0]

    self.assertTrue(torch.isfinite(guided).all().item())
    self.assertFalse(torch.allclose(base, guided))
    self.assertTrue(torch.equal(base, restored))
    self.assertEqual(
      tr.transformer_blocks[0].attn1.to_q.weight.dtype, torch.float8_e4m3fn
    )


class AnimaGuardsAndDecensorTests(unittest.TestCase):
  def _make_fake_anima_dir(self, root: Path):
    for sub in ANIMA_SUBDIRS:
      d = root / sub
      d.mkdir(parents=True, exist_ok=True)
      (d / f"{sub}.safetensors").write_bytes(b"stub")

  def test_anima_availability_guards_cuda_deps_and_files(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      with (
        patch.object(anima, "anima_dependencies_available", return_value=False),
        patch.object(anima, "anima_cuda_available", return_value=True),
      ):
        ok, reason = anima_availability(base_dir=root)
        self.assertFalse(ok)
        self.assertIn("dependencies", reason)

      with (
        patch.object(anima, "anima_dependencies_available", return_value=True),
        patch.object(anima, "anima_cuda_available", return_value=False),
      ):
        ok, reason = anima_availability(base_dir=root)
        self.assertFalse(ok)
        self.assertIn("CUDA", reason)

      with (
        patch.object(anima, "anima_dependencies_available", return_value=True),
        patch.object(anima, "anima_cuda_available", return_value=True),
      ):
        ok, reason = anima_availability(base_dir=root)
        self.assertFalse(ok)
        self.assertIn("missing", reason.lower())

        self._make_fake_anima_dir(root)
        ok, _ = anima_availability(base_dir=root)
        self.assertTrue(ok)

  def test_anima_rejected_in_mosaic_mode_and_without_cuda(self):
    with self.assertRaisesRegex(ValueError, "bar mode"):
      create_inpainter("anima", is_mosaic=True)

    with (
      patch.object(anima, "anima_dependencies_available", return_value=True),
      patch.object(anima, "anima_cuda_available", return_value=False),
      self.assertRaisesRegex(RuntimeError, "CUDA"),
    ):
      AnimaInpainter()

    with (
      patch.object(
        sys, "argv", [sys.argv[0], "--model", "anima", "--is_mosaic", "true"]
      ),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      with self.assertRaises(SystemExit) as ctx:
        config.get_args(cli=True)
      self.assertEqual(ctx.exception.code, 2)

    with (
      patch.object(sys, "argv", [sys.argv[0], "--model", "anima"]),
      patch.object(config, "anima_dependencies_available", return_value=True),
      patch.object(config, "anima_cuda_available", return_value=False),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      with self.assertRaises(SystemExit) as ctx:
        config.get_args(cli=True)
      self.assertEqual(ctx.exception.code, 2)

  def test_anima_variations_use_distinct_seeds_without_flip_and_support_cancel(self):
    class StubAnimaModel:
      batch_size = 1
      device = torch.device("cpu")
      fallback_reason = None

      def __init__(self, *, cancel_on_second_call=False, decensor_ref=None):
        self.calls = []
        self.cancel_on_second_call = cancel_on_second_call
        self.decensor_ref = decensor_ref

      def update_settings(self, settings):
        self.settings = settings

      def inpaint_crops(
        self,
        ori,
        marked,
        boxes,
        *,
        is_mosaic=False,
        regions=None,
        seed=None,
        cancel_check=None,
        progress_callback=None,
      ):
        self.calls.append(
          {
            "size": ori.size,
            "boxes": list(boxes),
            "seed": seed,
            "top_left_pixel": ori.getpixel((0, 0)),
          }
        )
        if progress_callback:
          progress_callback(1, 2, "Denoising step 1/2")
        if self.cancel_on_second_call and len(self.calls) == 2:
          if self.decensor_ref is not None:
            self.decensor_ref.request_cancel()
          if cancel_check and cancel_check():
            raise AnimaCancelledError("Cancelled")
        if progress_callback:
          progress_callback(2, 2, "Denoising step 2/2")
        out = []
        for box in boxes:
          w, h = box[2] - box[0], box[3] - box[1]
          val = (seed or 0) % 200 + 20
          out.append(np.full((h, w, 3), val, dtype=np.uint8))
        return out

    # Asymmetric image (distinct top-left pixel) to confirm Anima does NOT flip variations.
    arr = np.full((128, 128, 3), (180, 160, 140), dtype=np.uint8)
    arr[0, 0] = (11, 22, 33)
    mask = np.zeros((128, 128), dtype=bool)
    mask[30:40, 20:40] = True
    mask[45:55, 50:70] = True
    ori = Image.fromarray(arr)
    colored = apply_mask_to_image(ori, mask)

    with tempfile.TemporaryDirectory() as tmp:
      with patch.object(sys, "argv", [sys.argv[0]]):
        decensor = Decensor()
      decensor.decensor_output_path = tmp
      decensor.model_name = "anima"
      decensor.is_mosaic = False
      decensor.variations = 2
      decensor.anima_settings = AnimaSettings(seed=100, steps=2)
      stub = StubAnimaModel()
      decensor.model = stub
      decensor.loaded_config = decensor._current_model_config_key()

      with contextlib.redirect_stdout(io.StringIO()):
        results, _ = decensor.decensor_single_image(ori, colored, "anima_test")

      self.assertEqual(len(results), 2)
      self.assertEqual([c["seed"] for c in stub.calls], [100, 101])
      # Both variations kept unflipped top-left pixel and grouped 2 regions into 1 pass.
      self.assertEqual(
        [c["top_left_pixel"] for c in stub.calls], [(11, 22, 33), (11, 22, 33)]
      )
      self.assertEqual([len(c["boxes"]) for c in stub.calls], [1, 1])
      # Distinct seeds produced distinct filled pixels while keeping unmasked pixels intact.
      res0 = np.asarray(results[0])
      res1 = np.asarray(results[1])
      np.testing.assert_array_equal(res0[~mask], arr[~mask])
      np.testing.assert_array_equal(res1[~mask], arr[~mask])
      self.assertFalse(np.array_equal(res0[mask], res1[mask]))

      # Verify cancellation mid-batch keeps completed output & skips cleanup & keeps model.
      in_dir = Path(tmp) / "in"
      out_dir = Path(tmp) / "out"
      in_dir.mkdir()
      out_dir.mkdir()
      colored.save(in_dir / "img1.png")
      colored.save(in_dir / "img2.png")

      decensor.decensor_input_path = str(in_dir)
      decensor.decensor_output_path = str(out_dir)
      decensor.variations = 1
      decensor.clean_up_input_dirs = True
      cancel_stub = StubAnimaModel(cancel_on_second_call=True, decensor_ref=decensor)
      decensor.model = cancel_stub
      decensor.loaded_config = decensor._current_model_config_key()
      decensor.reset_cancel()

      with (
        contextlib.redirect_stdout(io.StringIO()),
        self.assertRaises((ProcessingCancelledError, AnimaCancelledError)),
      ):
        decensor.decensor_all_images_in_folder()

      # First image output kept, input dir NOT cleaned up, model still resident.
      self.assertTrue((out_dir / "img1 0.png").is_file())
      self.assertFalse((out_dir / "img2 0.png").exists())
      self.assertTrue((in_dir / "img1.png").is_file())
      self.assertIs(decensor.model, cancel_stub)


class AnimaGuiTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.app = QApplication.instance() or QApplication([])

  def test_anima_gui_disabled_when_unavailable_and_synced_when_available(self):
    # Case 1: Unavailable (no GPU) -> disabled with clear tooltip
    with (
      patch.object(MainWindow, "load_model", lambda self: None),
      patch.object(sys, "argv", [sys.argv[0]]),
      patch.object(
        model,
        "anima_availability",
        return_value=(False, "Anima requires a CUDA-capable NVIDIA GPU."),
      ),
      patch(
        "main.anima_availability",
        return_value=(False, "Anima requires a CUDA-capable NVIDIA GPU."),
      ),
      patch(
        "mask_editor.anima_availability",
        return_value=(False, "Anima requires a CUDA-capable NVIDIA GPU."),
      ),
    ):
      win = MainWindow()
      self.assertFalse(win.animaModelButton.isEnabled())
      self.assertIn("CUDA", win.animaModelButton.toolTip())
      self.assertFalse(win.animaSettingsPanel.isVisible())
      win.openMaskEditor()
      self.assertFalse(win.maskEditorWindow.animaRadio.isEnabled())
      win.close()

    # Case 2: Available -> 3-way sync, prompt/advanced sync, mosaic fallback, cancel wiring
    with (
      patch.object(MainWindow, "load_model", lambda self: None),
      patch.object(sys, "argv", [sys.argv[0]]),
      patch("main.anima_availability", return_value=(True, "Anima ready")),
      patch("mask_editor.anima_availability", return_value=(True, "Anima ready")),
    ):
      win = MainWindow()
      self.assertTrue(win.animaModelButton.isEnabled())
      win.openMaskEditor()
      editor = win.maskEditorWindow
      self.assertIsNotNone(editor)
      self.assertTrue(editor.animaRadio.isEnabled())

      # Select Anima in MainWindow and edit auxiliary prompt -> syncs to MaskEditor
      win.animaModelButton.setChecked(True)
      self.assertFalse(win.ttaCheckbox.isEnabled())
      self.assertTrue(win.animaSettingsPanel.isEnabled())
      self.assertTrue(editor.animaRadio.isChecked())
      self.assertEqual(
        win.animaSettingsPanel.negativePromptEdit.text(),
        ANIMA_DEFAULT_NEGATIVE_PROMPT,
      )

      win.animaSettingsPanel.positivePromptEdit.setText("1girl, @artist")
      win.animaSettingsPanel.stepsSpin.setValue(20)
      win.animaSettingsPanel.fp8Checkbox.setChecked(False)
      self.assertEqual(
        editor.animaSettingsPanel.positivePromptEdit.text(), "1girl, @artist"
      )
      self.assertEqual(editor.animaSettingsPanel.stepsSpin.value(), 20)
      self.assertFalse(editor.animaSettingsPanel.fp8Checkbox.isChecked())

      # Edit negative prompt & FP8 toggle in Editor -> syncs back to MainWindow
      editor.animaSettingsPanel.negativePromptEdit.setText("")
      editor.animaSettingsPanel.fp8Checkbox.setChecked(True)
      self.assertEqual(win.animaSettingsPanel.negativePromptEdit.text(), "")
      self.assertTrue(win.animaSettingsPanel.fp8Checkbox.isChecked())

      # Switch to Mosaic -> Anima disabled, falls back to PEPSI
      win.mosaicButton.setChecked(True)
      self.assertFalse(win.animaModelButton.isEnabled())
      self.assertTrue(win.pepsiModelButton.isChecked())
      self.assertFalse(editor.animaRadio.isEnabled())
      self.assertTrue(editor.pepsiRadio.isChecked())

      win.close()


if __name__ == "__main__":
  unittest.main()
