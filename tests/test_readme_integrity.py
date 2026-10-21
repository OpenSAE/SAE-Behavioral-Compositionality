"""Guard against accidental duplication or truncation of the public SAR READMEs."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize(
    "filename,title,headings,main,figure_map",
    [
        (
            "README.md",
            "# 🧩 Behavioral Compositionality of Sparse Autoencoder Representations",
            ["🎥 Overview","🎞️ 5-Minute Tutorial","✨ Highlights","📐 What SAR Measures","🔎 Comparison of Methods","🚀 Quickstart","📊 Main Results","🗺️ Figure Map","🙏 Appreciation","📚 Citation"],
            "## 📊 Main Results",
            "## 🗺️ Figure Map",
        ),
        (
            "README_cn.md",
            "# 稀疏自编码器表示的行为组合性",
            ["🎥 概览","🎞️ 5 分钟讲解","✨ 亮点","📐 SAR 衡量什么","🔎 方法对比","🚀 快速上手","📊 主要结果","🗺️ 图表索引","🙏 致谢"],
            "## 📊 主要结果",
            "## 🗺️ 图表索引",
        ),
    ],
)
def test_readme_layout_and_scientific_content(filename, title, headings, main, figure_map):
    text = (ROOT / filename).read_text(encoding="utf-8")

    # Catch a repeated preamble, dropped section or corrupted string substitution.
    assert text.splitlines().count(title) == 1
    assert re.findall(r"^## (.+)$", text, re.MULTILINE) == headings
    assert text.count("<details>") == text.count("</details>")
    assert text.count("## 🚀 ") == 1

    highlights = text.split("## ✨ ", 1)[1].split("## 📐 ", 1)[0]
    assert highlights.count("$`") == highlights.count("`$")
    assert r"$`\widehat{\mathrm{SAR}}_1`$" in highlights
    assert len(re.findall(r"^- ", highlights, re.MULTILINE)) == 7

    results = text.split(main, 1)[1].split(figure_map, 1)[0]
    for figure in range(2, 9):
        image = f"assets/figures/fig{figure}.png"
        assert (ROOT / image).is_file()
        assert f'<img src="{image}"' in results

    for value in ("0.9782", "0.9959", "0.5874", "0.7790", "0.1444", "0.0606"):
        assert value in text
    assert "notebooks/SAR_Quickstart.ipynb" in text
