"""Build the Windows x64 GUI bundle and its distribution 7z archive."""

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
COMPRESSION_LEVEL = 7
BUNDLE_MODELS = ("bar", "mosaic")
BUNDLE_DIRECTORIES = (
  "decensor_input",
  "decensor_input_original",
  "decensor_output",
  "models/lama",
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


def main():
  dist = ROOT / "dist"
  if (dist / NAME).exists() or (dist / ARCHIVE).exists():
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
        "src/main.py",
      ],
      cwd=ROOT,
      env={**os.environ, "PYINSTALLER_CONFIG_DIR": str(stage / "cache")},
      check=True,
    )
    bundle = stage / NAME
    for directory in BUNDLE_DIRECTORIES:
      (bundle / directory).mkdir(parents=True, exist_ok=True)
    copy_notices(bundle)

    nanazip = find_nanazip()
    if nanazip is not None:
      print(
        f"Compressing {ARCHIVE} (level {COMPRESSION_LEVEL})...",
        flush=True,
      )
      create_7z_archive(nanazip, stage, stage / ARCHIVE)
      bundle.rename(dist / NAME)
      (stage / ARCHIVE).rename(dist / ARCHIVE)
      print(f"Distribution is ready at {dist / ARCHIVE}")
    else:
      bundle.rename(dist / NAME)
      print(
        "Warning: NanaZipC executable not found. Skipped creating "
        f"{ARCHIVE}; bundle folder is ready at {dist / NAME}",
        file=sys.stderr,
      )


if __name__ == "__main__":
  main()
