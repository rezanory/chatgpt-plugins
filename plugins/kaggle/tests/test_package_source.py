import zipfile
from pathlib import Path

from chatgpt_plugin_kaggle.package_source import package_source


def test_source_packager_excludes_common_secrets(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("print('ok')")
    (source / ".env").write_text("SECRET=x")
    (source / "private.key").write_text("key")
    output = tmp_path / "source.zip"
    package_source(source, output)
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "app.py" in names
    assert ".env" not in names
    assert "private.key" not in names
