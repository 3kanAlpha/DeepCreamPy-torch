import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_output(path):
  with NamedTemporaryFile(
    dir=path.parent, prefix=f".{path.name}.", delete=False
  ) as stream:
    temporary = Path(stream.name)
  try:
    yield temporary
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(path):
  # Windows removes this lock even when the process is forcibly terminated.
  delete_on_close = getattr(os, "O_TEMPORARY", 0)
  try:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | delete_on_close)
  except (FileExistsError, PermissionError) as error:
    if path.exists():
      raise ValueError(f"Output is locked: {path}") from error
    raise
  try:
    yield
  finally:
    os.close(descriptor)
    if not delete_on_close:
      path.unlink()
