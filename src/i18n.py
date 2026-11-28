"""External JSON-based internationalization (i18n) support for DeepCreamPy GUI."""

from __future__ import annotations

import json
import locale
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_FALLBACK_LANGUAGE = "en"
SETTINGS_FILENAME = "ui_settings.json"


def app_root() -> Path:
  """Return the editable application root (beside exe when frozen, else repo root)."""
  if getattr(sys, "frozen", False):
    return Path(sys.executable).resolve().parent
  return Path(__file__).resolve().parent.parent


def default_lang_dir() -> Path:
  return app_root() / "lang"


def default_settings_path() -> Path:
  return app_root() / SETTINGS_FILENAME


def _flatten_mapping(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
  """Flatten nested dictionaries into dotted keys while preserving flat keys."""
  flat: dict[str, str] = {}
  for raw_key, value in data.items():
    key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
    if isinstance(value, dict):
      flat.update(_flatten_mapping(value, key))
    elif isinstance(value, str):
      flat[key] = value
  return flat


def load_lang_file(path: Path) -> dict[str, str]:
  """Read and flatten a JSON language file; return an empty dict on any error."""
  try:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    return {}
  if not isinstance(raw, dict):
    return {}
  return _flatten_mapping(raw)


def _normalize_locale_tag(tag: str) -> str:
  return tag.strip().replace("-", "_")


def _get_system_locale_name() -> str:
  try:
    from PySide6.QtCore import QLocale

    qt_name = QLocale.system().name()
    if qt_name:
      return qt_name
  except (ImportError, RuntimeError):
    return _get_stdlib_locale_name()
  return _get_stdlib_locale_name()


def _get_stdlib_locale_name() -> str:
  loc = locale.getlocale()[0]
  if loc:
    return loc
  for env_var in ("LC_ALL", "LC_MESSAGES", "LANG"):
    val = os.environ.get(env_var)
    if val:
      return val.split(".", 1)[0]
  return DEFAULT_FALLBACK_LANGUAGE


def detect_system_language(
  available_codes: Sequence[str],
  *,
  locale_name: str | None = None,
  fallback: str = DEFAULT_FALLBACK_LANGUAGE,
) -> str:
  """Select the best matching language code for the current system locale."""
  if not available_codes:
    return fallback

  normalized_map = {
    _normalize_locale_tag(code).casefold(): code for code in available_codes
  }
  raw_locale = (
    _get_system_locale_name() if locale_name is None else str(locale_name or "")
  )
  normalized_locale = _normalize_locale_tag(raw_locale).casefold()

  if normalized_locale in normalized_map:
    return normalized_map[normalized_locale]

  base = normalized_locale.split("_", 1)[0]
  if base and base in normalized_map:
    return normalized_map[base]

  if fallback.casefold() in normalized_map:
    return normalized_map[fallback.casefold()]

  return available_codes[0]


def load_saved_language(settings_path: Path | None = None) -> str | None:
  path = default_settings_path() if settings_path is None else Path(settings_path)
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    return None
  if not isinstance(data, dict):
    return None
  lang = data.get("language")
  if isinstance(lang, str) and lang.strip():
    return lang.strip()
  return None


def save_language(code: str, settings_path: Path | None = None) -> bool:
  path = default_settings_path() if settings_path is None else Path(settings_path)
  existing: dict[str, Any] = {}
  try:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
      existing = loaded
  except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    existing = {}

  existing["language"] = code
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
      json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
      encoding="utf-8",
    )
    return True
  except OSError:
    return False


class Translator:
  """Manages external JSON language files, fallback lookup, and live UI updates."""

  def __init__(
    self,
    lang_dir: Path | None = None,
    settings_path: Path | None = None,
    *,
    fallback_code: str = DEFAULT_FALLBACK_LANGUAGE,
  ) -> None:
    self.lang_dir = default_lang_dir() if lang_dir is None else Path(lang_dir)
    self.settings_path = (
      default_settings_path() if settings_path is None else Path(settings_path)
    )
    self.fallback_code = fallback_code
    self.current_code = fallback_code
    self._fallback_strings: dict[str, str] = {}
    self._strings: dict[str, str] = {}
    self._listeners: list[Callable[[str], None]] = []
    self._qt_translator: Any = None
    self._reload_current()

  def available_languages(self) -> list[tuple[str, str]]:
    """Discover all ``*.json`` language files and return ``[(code, display_name)]``."""
    if not self.lang_dir.is_dir():
      return []
    entries: list[tuple[str, str]] = []
    for path in sorted(self.lang_dir.glob("*.json"), key=lambda p: p.stem.casefold()):
      if not path.is_file():
        continue
      code = path.stem
      strings = load_lang_file(path)
      display_name = strings.get("_meta.name") or code
      entries.append((code, display_name))
    # Keep fallback language first for predictable UI ordering.
    entries.sort(
      key=lambda item: (
        0 if item[0].casefold() == self.fallback_code.casefold() else 1,
        item[0].casefold(),
      )
    )
    return entries

  def _resolve_code(self, requested: str) -> str:
    available = [code for code, _ in self.available_languages()]
    lookup = {_normalize_locale_tag(c).casefold(): c for c in available}
    norm = _normalize_locale_tag(requested).casefold()
    if norm in lookup:
      return lookup[norm]
    base = norm.split("_", 1)[0]
    if base in lookup:
      return lookup[base]
    return requested

  def _reload_current(self) -> None:
    self._fallback_strings = load_lang_file(
      self.lang_dir / f"{self.fallback_code}.json"
    )
    if self.current_code.casefold() == self.fallback_code.casefold():
      self._strings = dict(self._fallback_strings)
    else:
      self._strings = load_lang_file(self.lang_dir / f"{self.current_code}.json")

  def _install_qt_translator(self, code: str) -> None:
    try:
      from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
      from PySide6.QtWidgets import QApplication
    except (ImportError, RuntimeError):
      return

    app = QApplication.instance()
    if app is None:
      return

    if self._qt_translator is not None:
      app.removeTranslator(self._qt_translator)
      self._qt_translator = None

    translations_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if not translations_path:
      return

    translator = QTranslator(app)
    qt_locale = QLocale(code)
    if translator.load(qt_locale, "qtbase", "_", translations_path):
      app.installTranslator(translator)
      self._qt_translator = translator

  def add_listener(self, callback: Callable[[str], None]) -> None:
    if callback not in self._listeners:
      self._listeners.append(callback)

  def remove_listener(self, callback: Callable[[str], None]) -> None:
    if callback in self._listeners:
      self._listeners.remove(callback)

  def set_language(self, code: str, *, persist: bool = False) -> str:
    resolved = self._resolve_code(code)
    self.current_code = resolved
    self._reload_current()
    if persist:
      save_language(resolved, self.settings_path)
    self._install_qt_translator(resolved)
    for callback in list(self._listeners):
      callback(resolved)
    return resolved

  def init_language(
    self,
    *,
    cli_lang: str | None = None,
    locale_name: str | None = None,
  ) -> str:
    """Resolve initial language from CLI override -> saved settings -> OS locale."""
    available_codes = [code for code, _ in self.available_languages()]
    available_lookup = {_normalize_locale_tag(c).casefold(): c for c in available_codes}

    chosen: str | None = None
    if cli_lang:
      chosen = self._resolve_code(cli_lang)
    else:
      saved = load_saved_language(self.settings_path)
      if saved:
        norm_saved = _normalize_locale_tag(saved).casefold()
        if norm_saved in available_lookup:
          chosen = available_lookup[norm_saved]

    if chosen is None:
      chosen = detect_system_language(
        available_codes,
        locale_name=locale_name,
        fallback=self.fallback_code,
      )

    return self.set_language(chosen, persist=False)

  def translate(self, key: str, **kwargs: Any) -> str:
    template = self._strings.get(key)
    if template is None:
      template = self._fallback_strings.get(key)
    if template is None:
      return key
    if not kwargs:
      return template
    try:
      return template.format(**kwargs)
    except (KeyError, ValueError, IndexError):
      fallback_template = self._fallback_strings.get(key)
      if fallback_template is not None and fallback_template != template:
        try:
          return fallback_template.format(**kwargs)
        except (KeyError, ValueError, IndexError):
          pass
      return template


i18n = Translator()


def t(key: str, **kwargs: Any) -> str:
  """Translate a message key using the active global Translator."""
  return i18n.translate(key, **kwargs)
