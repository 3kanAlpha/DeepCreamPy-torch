import json
import os
import string
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))

from PIL import Image
from PySide6.QtWidgets import QApplication

import i18n as i18n_mod
from i18n import (
  Translator,
  default_lang_dir,
  default_settings_path,
  detect_system_language,
  load_lang_file,
  load_saved_language,
  save_language,
)
from main import MainWindow, _extract_cli_lang


def _placeholder_names(template: str) -> set[str]:
  formatter = string.Formatter()
  return {
    field_name
    for _, field_name, _, _ in formatter.parse(template)
    if field_name is not None
  }


class LangFilesIntegrityTests(unittest.TestCase):
  def test_en_and_ja_keys_and_placeholders_match(self):
    en_path = ROOT / "lang" / "en.json"
    ja_path = ROOT / "lang" / "ja.json"
    self.assertTrue(en_path.is_file())
    self.assertTrue(ja_path.is_file())

    en = load_lang_file(en_path)
    ja = load_lang_file(ja_path)
    self.assertEqual(set(en.keys()), set(ja.keys()))
    self.assertGreater(len(en), 50)

    for key, en_val in en.items():
      ja_val = ja[key]
      self.assertTrue(en_val.strip(), f"Empty English string for {key}")
      self.assertTrue(ja_val.strip(), f"Empty Japanese string for {key}")
      self.assertEqual(
        _placeholder_names(en_val),
        _placeholder_names(ja_val),
        f"Placeholder mismatch for key {key}",
      )


class TranslatorUnitTests(unittest.TestCase):
  def test_nested_flat_discovery_fallback_and_live_reload(self):
    with tempfile.TemporaryDirectory() as tmp:
      lang_dir = Path(tmp) / "lang"
      lang_dir.mkdir()
      settings_path = Path(tmp) / "ui_settings.json"

      (lang_dir / "en.json").write_text(
        json.dumps(
          {
            "_meta": {"name": "English"},
            "main": {"hello": "Hello {name}", "only_en": "English Only"},
          }
        ),
        encoding="utf-8",
      )
      (lang_dir / "ja.json").write_text(
        json.dumps(
          {
            "_meta": {"name": "日本語"},
            "main.hello": "こんにちは {name}",
          },
          ensure_ascii=False,
        ),
        encoding="utf-8",
      )
      (lang_dir / "broken.json").write_text("{not valid json", encoding="utf-8")

      tr = Translator(lang_dir=lang_dir, settings_path=settings_path)
      langs = tr.available_languages()
      self.assertEqual(langs[0], ("en", "English"))
      self.assertIn(("ja", "日本語"), langs)
      self.assertIn(("broken", "broken"), langs)

      tr.set_language("ja")
      self.assertEqual(tr.translate("main.hello", name="DCP"), "こんにちは DCP")
      # Missing key in ja falls back to en.
      self.assertEqual(tr.translate("main.only_en"), "English Only")
      # Missing key in both falls back to key name itself.
      self.assertEqual(tr.translate("main.unknown_key"), "main.unknown_key")

      # Broken JSON file safely falls back to en.
      tr.set_language("broken")
      self.assertEqual(tr.translate("main.hello", name="DCP"), "Hello DCP")

      # Broken format placeholder in ja safely falls back to en formatting.
      (lang_dir / "ja.json").write_text(
        json.dumps(
          {
            "_meta": {"name": "日本語"},
            "main": {"hello": "こんにちは {bad_placeholder}"},
          },
          ensure_ascii=False,
        ),
        encoding="utf-8",
      )
      tr.set_language("ja")
      self.assertEqual(tr.translate("main.hello", name="DCP"), "Hello DCP")

      # External edit on disk is picked up on re-selecting the language.
      (lang_dir / "ja.json").write_text(
        json.dumps(
          {
            "_meta": {"name": "日本語"},
            "main": {"hello": "ようこそ {name}"},
          },
          ensure_ascii=False,
        ),
        encoding="utf-8",
      )
      tr.set_language("ja")
      self.assertEqual(tr.translate("main.hello", name="DCP"), "ようこそ DCP")

  def test_locale_detection_persistence_and_priority(self):
    self.assertEqual(detect_system_language(["en", "ja"], locale_name="ja_JP"), "ja")
    self.assertEqual(detect_system_language(["en", "ja"], locale_name="ja-JP"), "ja")
    self.assertEqual(detect_system_language(["en", "ja"], locale_name="fr_FR"), "en")
    self.assertEqual(
      detect_system_language(["en", "zh_CN"], locale_name="zh-CN"), "zh_CN"
    )

    with tempfile.TemporaryDirectory() as tmp:
      lang_dir = Path(tmp) / "lang"
      lang_dir.mkdir()
      for code, name in (("en", "English"), ("ja", "日本語")):
        (lang_dir / f"{code}.json").write_text(
          json.dumps({"_meta": {"name": name}}, ensure_ascii=False),
          encoding="utf-8",
        )
      settings_path = Path(tmp) / "ui_settings.json"
      settings_path.write_text(json.dumps({"other_key": 123}), encoding="utf-8")

      # Auto-detect when no saved language exists.
      tr = Translator(lang_dir=lang_dir, settings_path=settings_path)
      self.assertEqual(tr.init_language(locale_name="ja_JP"), "ja")

      # Persist preserves other settings keys.
      tr.set_language("ja", persist=True)
      self.assertEqual(load_saved_language(settings_path), "ja")
      saved_raw = json.loads(settings_path.read_text(encoding="utf-8"))
      self.assertEqual(saved_raw["other_key"], 123)

      # Saved setting overrides OS locale.
      tr2 = Translator(lang_dir=lang_dir, settings_path=settings_path)
      self.assertEqual(tr2.init_language(locale_name="en_US"), "ja")

      # CLI --lang overrides saved setting.
      self.assertEqual(tr2.init_language(cli_lang="en", locale_name="ja_JP"), "en")
      self.assertEqual(_extract_cli_lang(["--lang", "ja", "--variations", "2"]), "ja")

      # Saving directly works even when file does not exist yet.
      fresh_settings = Path(tmp) / "sub" / "ui_settings.json"
      self.assertTrue(save_language("ja", fresh_settings))
      self.assertEqual(load_saved_language(fresh_settings), "ja")

  def test_frozen_paths_resolve_beside_executable(self):
    with tempfile.TemporaryDirectory() as tmp:
      fake_exe = Path(tmp) / "DeepCreamPy" / "DeepCreamPy.exe"
      fake_exe.parent.mkdir(parents=True)
      fake_exe.write_bytes(b"")
      with (
        patch.object(sys, "frozen", True, create=True),
        patch.object(sys, "executable", str(fake_exe)),
      ):
        self.assertEqual(default_lang_dir(), fake_exe.resolve().parent / "lang")
        self.assertEqual(
          default_settings_path(),
          fake_exe.resolve().parent / "ui_settings.json",
        )


class GuiLiveI18nTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.app = QApplication.instance() or QApplication([])

  def test_live_language_switch_updates_main_editor_and_anima_panel(self):
    with tempfile.TemporaryDirectory() as tmp:
      settings_path = Path(tmp) / "ui_settings.json"
      orig_settings_path = i18n_mod.i18n.settings_path
      orig_code = i18n_mod.i18n.current_code
      i18n_mod.i18n.settings_path = settings_path
      i18n_mod.i18n.set_language("en", persist=False)

      try:
        with (
          patch.object(MainWindow, "load_model", lambda self: None),
          patch.object(sys, "argv", [sys.argv[0]]),
        ):
          win = MainWindow()
          win.openMaskEditor()
          editor = win.maskEditorWindow
          self.assertIsNotNone(editor)

          # Populate variation combo so retranslation of combo items is also verified.
          sample = Image.new("RGB", (32, 32), (100, 100, 100))
          editor._set_loaded_image(sample, "sample", source_path=None)
          editor.on_decensor_succeeded([sample, sample], ["out/a 0.png", "out/a 1.png"])

          self.assertEqual(win.censorTypeGroupBox.title(), "Censor Type")
          self.assertEqual(win.decensorButton.text(), "Decensor Your Images")
          self.assertEqual(editor.openButton.text(), "Open Image...")
          self.assertEqual(editor.variationCombo.itemText(0), "Variation #0")
          self.assertEqual(win.animaSettingsPanel.title(), "Anima Settings")

          # Switch language via MainWindow combo box -> live updates everywhere & persists.
          ja_idx = win.languageCombo.findData("ja")
          self.assertGreaterEqual(ja_idx, 0)
          win.languageCombo.setCurrentIndex(ja_idx)

          self.assertEqual(load_saved_language(settings_path), "ja")
          self.assertEqual(win.censorTypeGroupBox.title(), "修正タイプ")
          self.assertEqual(win.decensorButton.text(), "画像を修正除去")
          self.assertEqual(editor.openButton.text(), "画像を開く...")
          self.assertEqual(editor.detectGroup.title(), "自動検出（バー）")
          self.assertEqual(editor.variationCombo.itemText(0), "バリエーション #0")
          self.assertEqual(win.animaSettingsPanel.title(), "Anima 設定")
          self.assertEqual(editor.animaSettingsPanel.title(), "Anima 設定")

          # Switch back to English via combo box.
          en_idx = win.languageCombo.findData("en")
          self.assertGreaterEqual(en_idx, 0)
          win.languageCombo.setCurrentIndex(en_idx)

          self.assertEqual(win.censorTypeGroupBox.title(), "Censor Type")
          self.assertEqual(editor.openButton.text(), "Open Image...")
          self.assertEqual(editor.variationCombo.itemText(0), "Variation #0")

          win.close()
      finally:
        i18n_mod.i18n.settings_path = orig_settings_path
        i18n_mod.i18n.set_language(orig_code, persist=False)


if __name__ == "__main__":
  unittest.main()
