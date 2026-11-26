"""Build the Windows x64 GUI bundle and its distribution ZIP."""

import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import distributions
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
NAME = "DeepCreamPy"
ARCHIVE = f"{NAME}-Windows-x64.zip"


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
      f"Move the existing {NAME} bundle and ZIP out of {dist} before rebuilding."
    )

  model_data = []
  for mode in ("bar", "mosaic"):
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
    for directory in ("decensor_input", "decensor_input_original", "decensor_output"):
      (bundle / directory).mkdir()
    copy_notices(bundle)

    print(f"Compressing {ARCHIVE}...", flush=True)
    with ZipFile(stage / ARCHIVE, "w", ZIP_DEFLATED, compresslevel=1) as archive:
      for file in sorted(bundle.rglob("*")):
        archive.write(file, file.relative_to(stage))
    bundle.rename(dist / NAME)
    (stage / ARCHIVE).rename(dist / ARCHIVE)
  print(f"Distribution is ready at {dist / ARCHIVE}")


if __name__ == "__main__":
  main()
