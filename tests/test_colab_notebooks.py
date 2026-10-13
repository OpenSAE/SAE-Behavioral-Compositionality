"""No network needed: validate new Colab notebooks and run their synthetic cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "notebooks" / f"{name}.ipynb"
    book = json.loads(path.read_text(encoding="utf-8"))
    assert book["nbformat"] == 4
    assert book["cells"]
    for cell in book["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
    return book


def _cell_source(book, prefix):
    for cell in book["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and prefix in source:
            return source
    raise AssertionError(f"code cell containing {prefix!r} not found")


def test_quickstart_synthetic_cells_execute():
    book = _load("SAR_Quickstart")
    env = {"__name__": "__main__"}
    exec(compile(_cell_source(book, "IS_COLAB ="), "setup", "exec"), env)
    exec(compile(_cell_source(book, "# A transparent, model-free"), "synthetic", "exec"), env)
    assert env["r"]["sar1"] > 0.01
    assert env["r"]["sar2"] < 1e-16


def test_custom_model_synthetic_cells_execute():
    book = _load("Use_Your_Own_SAE")
    env = {"__name__": "__main__"}
    exec(compile(_cell_source(book, "IS_COLAB ="), "setup", "exec"), env)
    exec(compile(_cell_source(book, "# EXAMPLE: an SAE"), "custom", "exec"), env)
    assert env["custom_result"].metrics["sar1"] > 0.01
    assert env["custom_result"].metrics["sar2"] < 1e-16


def test_colab_links_and_no_rewriting_existing_content():
    for name in ("SAR_Quickstart", "Use_Your_Own_SAE"):
        b = _load(name)
        text = "\n".join("".join(c["source"]) for c in b["cells"])
        assert f"/notebooks/{name}.ipynb" in text
        assert "github.com/OpenSAE/SAE-Behavioral-Compositionality.git" in text
    for lang in ("README.md", "README_cn.md"):
        text = (ROOT / lang).read_text(encoding="utf-8")
        assert "notebooks/SAR_Quickstart.ipynb" in text
        assert "notebooks/Use_Your_Own_SAE.ipynb" in text
