import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))

import build_windows
import model


class BuildWindowsTests(unittest.TestCase):
  def _prepare_fake_repo(
    self,
    root: Path,
    *,
    include_lama_model: bool = True,
    include_anima_assets: bool = False,
  ):
    (root / "LICENSE.md").write_text("license", encoding="utf-8")
    lang_dir = root / "lang"
    lang_dir.mkdir()
    (lang_dir / "en.json").write_text(
      '{"_meta": {"name": "English"}}', encoding="utf-8"
    )
    (lang_dir / "ja.json").write_text('{"_meta": {"name": "日本語"}}', encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "ACKNOWLEDGEMENTS.md").write_text("ack", encoding="utf-8")
    (docs / "INSTALLATION_BINARY.md").write_text("install", encoding="utf-8")
    for mode in ("bar", "mosaic"):
      model_dir = root / "models" / mode
      model_dir.mkdir(parents=True)
      (model_dir / "inpaint.pt").write_bytes(b"weights")
    if include_lama_model:
      lama_dir = root / "models" / "lama"
      lama_dir.mkdir(parents=True)
      (lama_dir / "inpaint.pt").write_bytes(b"lama-weights")
    if include_anima_assets:
      for tok in ("qwen3", "t5"):
        tok_dir = root / "assets" / "anima_tokenizers" / tok
        tok_dir.mkdir(parents=True)
        (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
      for sub in ("diffusion", "text_encoder", "vae", "controlnet"):
        wdir = root / "models" / "anima" / sub
        wdir.mkdir(parents=True)
        (wdir / f"{sub}.safetensors").write_bytes(b"anima-weights")

  def test_build_creates_7z_level7_and_excludes_lama_weights(self):
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp)
      self._prepare_fake_repo(repo, include_lama_model=True)
      pyinstaller_cmds = []
      seven_zip_cmds = []

      def fake_run(cmd, **kwargs):
        if "-m" in cmd and "PyInstaller" in cmd:
          pyinstaller_cmds.append(cmd)
          stage = Path(cmd[cmd.index("--distpath") + 1])
          bundle = stage / build_windows.NAME
          bundle.mkdir(parents=True)
          (bundle / f"{build_windows.NAME}.exe").write_bytes(b"exe")
        else:
          seven_zip_cmds.append((cmd, kwargs))
          archive_path = Path(cmd[cmd.index("-mx=7") + 1])
          archive_path.write_bytes(b"7z-archive")
        return subprocess.CompletedProcess(cmd, 0)

      with (
        patch.object(build_windows, "ROOT", repo),
        patch.object(build_windows, "distributions", return_value=[]),
        patch.object(build_windows, "find_nanazip", return_value="NanaZipC"),
        patch.object(build_windows.subprocess, "run", side_effect=fake_run),
      ):
        build_windows.main()

      self.assertEqual(len(pyinstaller_cmds), 1)
      add_data_args = [
        pyinstaller_cmds[0][i + 1]
        for i, arg in enumerate(pyinstaller_cmds[0])
        if arg == "--add-data"
      ]
      self.assertEqual(len(add_data_args), 2)
      self.assertTrue(any(arg.endswith(":models/bar") for arg in add_data_args))
      self.assertTrue(any(arg.endswith(":models/mosaic") for arg in add_data_args))
      self.assertFalse(any("lama" in arg.lower() for arg in add_data_args))

      self.assertEqual(len(seven_zip_cmds), 1)
      sz_cmd, sz_kwargs = seven_zip_cmds[0]
      self.assertEqual(sz_cmd[:4], ["NanaZipC", "a", "-t7z", "-mx=7"])
      self.assertEqual(sz_cmd[5], build_windows.NAME)
      self.assertTrue(sz_kwargs["check"])

      bundle_dir = repo / "dist" / build_windows.NAME
      self.assertTrue(bundle_dir.is_dir())
      self.assertTrue((bundle_dir / "lang" / "en.json").is_file())
      self.assertTrue((bundle_dir / "lang" / "ja.json").is_file())
      self.assertTrue((bundle_dir / "models" / "lama").is_dir())
      self.assertFalse((bundle_dir / "models" / "lama" / "inpaint.pt").exists())
      self.assertFalse((bundle_dir / "models" / "anima").exists())
      self.assertIn("--exclude-module", pyinstaller_cmds[0])
      self.assertTrue((repo / "dist" / build_windows.ARCHIVE).is_file())

  def test_build_with_anima_bundles_tokenizers_and_excludes_anima_weights(self):
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp)
      self._prepare_fake_repo(repo, include_lama_model=False, include_anima_assets=True)
      pyinstaller_cmds = []
      seven_zip_cmds = []

      def fake_run(cmd, **kwargs):
        if "-m" in cmd and "PyInstaller" in cmd:
          pyinstaller_cmds.append(cmd)
          stage = Path(cmd[cmd.index("--distpath") + 1])
          bundle = stage / build_windows.NAME
          bundle.mkdir(parents=True)
          (bundle / f"{build_windows.NAME}.exe").write_bytes(b"exe")
        else:
          seven_zip_cmds.append((cmd, kwargs))
          archive_path = Path(cmd[cmd.index("-mx=7") + 1])
          archive_path.write_bytes(b"7z-archive")
        return subprocess.CompletedProcess(cmd, 0)

      with (
        patch.object(build_windows, "ROOT", repo),
        patch.object(build_windows, "distributions", return_value=[]),
        patch.object(build_windows, "find_nanazip", return_value="NanaZipC"),
        patch.object(build_windows.importlib.util, "find_spec", return_value=object()),
        patch.object(build_windows.subprocess, "run", side_effect=fake_run),
      ):
        build_windows.main(["--with-anima"])

      self.assertEqual(len(pyinstaller_cmds), 1)
      add_data_args = [
        pyinstaller_cmds[0][i + 1]
        for i, arg in enumerate(pyinstaller_cmds[0])
        if arg == "--add-data"
      ]
      self.assertEqual(len(add_data_args), 4)
      self.assertTrue(
        any(arg.endswith(":assets/anima_tokenizers/qwen3") for arg in add_data_args)
      )
      self.assertTrue(
        any(arg.endswith(":assets/anima_tokenizers/t5") for arg in add_data_args)
      )
      self.assertFalse(any("safetensors" in arg.lower() for arg in add_data_args))
      self.assertNotIn("--exclude-module", pyinstaller_cmds[0])

      bundle_dir = repo / "dist" / build_windows.NAME
      for sub in ("diffusion", "text_encoder", "vae", "controlnet"):
        self.assertTrue((bundle_dir / "models" / "anima" / sub).is_dir())
        self.assertEqual(list((bundle_dir / "models" / "anima" / sub).iterdir()), [])
      self.assertTrue((repo / "dist" / build_windows.ANIMA_ARCHIVE).is_file())

  def test_build_warns_and_keeps_folder_when_nanazip_missing(self):
    with tempfile.TemporaryDirectory() as tmp:
      repo = Path(tmp)
      self._prepare_fake_repo(repo, include_lama_model=False)

      def fake_run(cmd, **kwargs):
        stage = Path(cmd[cmd.index("--distpath") + 1])
        bundle = stage / build_windows.NAME
        bundle.mkdir(parents=True)
        (bundle / f"{build_windows.NAME}.exe").write_bytes(b"exe")
        return subprocess.CompletedProcess(cmd, 0)

      stderr = io.StringIO()
      with (
        patch.object(build_windows, "ROOT", repo),
        patch.object(build_windows, "distributions", return_value=[]),
        patch.object(build_windows, "find_nanazip", return_value=None),
        patch.object(build_windows.subprocess, "run", side_effect=fake_run),
        redirect_stderr(stderr),
      ):
        build_windows.main()

      self.assertIn("NanaZipC executable not found", stderr.getvalue())
      self.assertTrue((repo / "dist" / build_windows.NAME).is_dir())
      self.assertTrue((repo / "dist" / build_windows.NAME / "models" / "lama").is_dir())
      self.assertFalse((repo / "dist" / build_windows.ARCHIVE).exists())

  def test_create_7z_archive_real_if_nanazip_available(self):
    nanazip = build_windows.find_nanazip()
    if nanazip is None:
      self.skipTest("NanaZipC executable is not available in this environment")
    with tempfile.TemporaryDirectory() as tmp:
      stage = Path(tmp)
      bundle = stage / build_windows.NAME
      (bundle / "models" / "lama").mkdir(parents=True)
      (bundle / "hello.txt").write_text("deepcreamy", encoding="utf-8")
      archive_path = stage / build_windows.ARCHIVE

      build_windows.create_7z_archive(nanazip, stage, archive_path)
      self.assertTrue(archive_path.is_file())

      listing = subprocess.run(
        [nanazip, "l", "-slt", "-sccUTF-8", str(archive_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
      ).stdout.replace("\\", "/")
      self.assertIn(f"Path = {build_windows.NAME}/hello.txt", listing)
      self.assertIn(f"Path = {build_windows.NAME}/models/lama", listing)


class FrozenLamaPathTests(unittest.TestCase):
  def test_lama_weights_path_resolves_beside_executable_when_frozen(self):
    with tempfile.TemporaryDirectory() as tmp:
      fake_exe = Path(tmp) / "DeepCreamPy" / "DeepCreamPy.exe"
      fake_exe.parent.mkdir(parents=True)
      fake_exe.write_bytes(b"")
      with (
        patch.object(sys, "frozen", True, create=True),
        patch.object(sys, "executable", str(fake_exe)),
      ):
        reloaded = importlib.reload(model)
        try:
          expected = fake_exe.resolve().parent / "models" / "lama" / "inpaint.pt"
          self.assertEqual(reloaded.LAMA_WEIGHTS_PATH, expected)
          self.assertFalse(reloaded.lama_weights_available())
          expected.parent.mkdir(parents=True, exist_ok=True)
          expected.write_bytes(b"stub")
          self.assertTrue(reloaded.lama_weights_available())
        finally:
          pass
      # Restore non-frozen module constants after reload.
      importlib.reload(model)


if __name__ == "__main__":
  unittest.main()
