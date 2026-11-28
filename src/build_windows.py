"""Build the Windows x64 GUI bundle and its distribution 7z archive."""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DeepCreamPy"
ARCHIVE = f"{NAME}-Windows-x64.7z"
ANIMA_ARCHIVE = f"{NAME}-Anima-Windows-x64.7z"
COMPRESSION_LEVEL = 7
BUNDLE_MODELS = ("bar", "mosaic")
BUNDLE_DIRECTORIES = (
  "decensor_input",
  "decensor_input_original",
  "decensor_output",
  "models/lama",
)
ANIMA_BUNDLE_DIRECTORIES = (
  "models/anima/diffusion",
  "models/anima/text_encoder",
  "models/anima/vae",
  "models/anima/controlnet",
)
ANIMA_REQUIRED_PACKAGES = (
  "diffusers",
  "transformers",
  "safetensors",
  "accelerate",
  "tokenizers",
)
ANIMA_EXCLUDED_MODULES = (
  "diffusers",
  "transformers",
  "safetensors",
  "accelerate",
  "tokenizers",
  "sentencepiece",
  "huggingface_hub",
)
ANIMA_HIDDEN_IMPORTS = (
  "diffusers.models.transformers.transformer_cosmos",
  "diffusers.models.condition_embedders.condition_embedder_anima",
  "diffusers.models.autoencoders.autoencoder_kl_qwenimage",
  "diffusers.schedulers.scheduling_flow_match_euler_discrete",
  "diffusers.hooks.group_offloading",
  "diffusers.loaders.single_file_utils",
  "transformers.models.qwen3.modeling_qwen3",
  "transformers.models.qwen3.configuration_qwen3",
  "transformers.models.qwen2.tokenization_qwen2",
  "transformers.models.t5.tokenization_t5_fast",
  "safetensors.torch",
)
ANIMA_COPY_METADATA = (
  "accelerate",
  "diffusers",
  "huggingface_hub",
  "safetensors",
  "tokenizers",
  "transformers",
)


def _is_working_nanazip(executable):
  try:
    result = subprocess.run(
      [executable, "i"],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
    )
  except OSError:
    return False
  return result.returncode == 0


def find_nanazip():
  candidates = []
  found = shutil.which("NanaZipC")
  if found:
    candidates.append(found)

  local_app_data = os.environ.get("LOCALAPPDATA")
  if local_app_data:
    candidate = Path(local_app_data) / "Microsoft" / "WindowsApps" / "NanaZipC.exe"
    if candidate.is_file():
      candidates.append(str(candidate))

  seen = set()
  for candidate in candidates:
    key = os.path.normcase(os.path.abspath(candidate))
    if key in seen:
      continue
    seen.add(key)
    if _is_working_nanazip(candidate):
      return candidate
  return None


def create_7z_archive(nanazip, stage, archive_path):
  subprocess.run(
    [
      nanazip,
      "a",
      "-t7z",
      f"-mx={COMPRESSION_LEVEL}",
      str(archive_path),
      NAME,
    ],
    cwd=stage,
    check=True,
  )


def copy_notices(bundle):
  for source in (
    "LICENSE.md",
    "docs/ACKNOWLEDGEMENTS.md",
    "docs/INSTALLATION_BINARY.md",
  ):
    shutil.copy2(ROOT / source, bundle)
  for distribution in distributions():
    for file in distribution.files or ():
      if not file.parts[0].endswith(".dist-info"):
        continue
      if any(
        part.lower().startswith(("license", "notice", "copying"))
        for part in file.parts[1:]
      ):
        target = bundle / "licenses" / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(distribution.locate_file(file), target)


def parse_args(argv):
  parser = argparse.ArgumentParser(
    description="Build the Windows x64 GUI bundle and its distribution 7z archive."
  )
  parser.add_argument(
    "--with-anima",
    action="store_true",
    default=False,
    help=(
      "include optional Anima dependencies (diffusers/transformers/safetensors) "
      "and offline tokenizers, and emit DeepCreamPy-Anima-Windows-x64.7z"
    ),
  )
  return parser.parse_args(argv)


def main(argv=None, *, with_anima=False):
  if argv is not None:
    parsed = parse_args(argv)
    with_anima = parsed.with_anima

  archive_name = ANIMA_ARCHIVE if with_anima else ARCHIVE
  dist = ROOT / "dist"
  if (dist / NAME).exists() or (dist / archive_name).exists():
    raise SystemExit(
      f"Move the existing {NAME} bundle and 7z archive out of {dist} before rebuilding."
    )

  model_data = []
  for mode in BUNDLE_MODELS:
    source = ROOT / "models" / mode / "inpaint.pt"
    if not source.is_file():
      raise SystemExit(
        f"Missing {source}. Run the model conversion in docs/INSTALLATION.md."
      )
    model_data.extend(("--add-data", f"{source}:models/{mode}"))

  extra_pyinstaller_args = []
  if with_anima:
    missing_pkgs = [
      pkg for pkg in ANIMA_REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None
    ]
    if missing_pkgs:
      raise SystemExit(
        f"Missing Anima build dependencies: {', '.join(missing_pkgs)}. "
        "Run with `uv run --locked --group windows-build --group anima python src/build_windows.py --with-anima`."
      )
    for tok_name in ("qwen3", "t5"):
      tok_dir = ROOT / "assets" / "anima_tokenizers" / tok_name
      if not (tok_dir / "tokenizer.json").is_file():
        raise SystemExit(
          f"Missing Anima offline tokenizer at {tok_dir / 'tokenizer.json'}."
        )
      model_data.extend(("--add-data", f"{tok_dir}:assets/anima_tokenizers/{tok_name}"))
    for mod in ANIMA_HIDDEN_IMPORTS:
      extra_pyinstaller_args.extend(("--hidden-import", mod))
    for dist_name in ANIMA_COPY_METADATA:
      extra_pyinstaller_args.extend(("--copy-metadata", dist_name))
  else:
    for mod in ANIMA_EXCLUDED_MODULES:
      extra_pyinstaller_args.extend(("--exclude-module", mod))

  build = ROOT / "build"
  build.mkdir(exist_ok=True)
  dist.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="windows-", dir=build) as temporary:
    stage = Path(temporary)
    subprocess.run(
      [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onedir",
        "--windowed",
        "--noupx",
        "--name",
        NAME,
        "--distpath",
        str(stage),
        "--specpath",
        str(stage),
        "--workpath",
        str(stage / "work"),
        *model_data,
        *extra_pyinstaller_args,
        "src/main.py",
      ],
      cwd=ROOT,
      env={**os.environ, "PYINSTALLER_CONFIG_DIR": str(stage / "cache")},
      check=True,
    )
    bundle = stage / NAME
    directories = (
      (*BUNDLE_DIRECTORIES, *ANIMA_BUNDLE_DIRECTORIES)
      if with_anima
      else BUNDLE_DIRECTORIES
    )
    for directory in directories:
      (bundle / directory).mkdir(parents=True, exist_ok=True)
    copy_notices(bundle)

    nanazip = find_nanazip()
    if nanazip is not None:
      print(
        f"Compressing {archive_name} (level {COMPRESSION_LEVEL})...",
        flush=True,
      )
      create_7z_archive(nanazip, stage, stage / archive_name)
      bundle.rename(dist / NAME)
      (stage / archive_name).rename(dist / archive_name)
      print(f"Distribution is ready at {dist / archive_name}")
    else:
      bundle.rename(dist / NAME)
      print(
        "Warning: NanaZipC executable not found. Skipped creating "
        f"{archive_name}; bundle folder is ready at {dist / NAME}",
        file=sys.stderr,
      )


if __name__ == "__main__":
  main(sys.argv[1:])
