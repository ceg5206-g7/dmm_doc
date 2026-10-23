from pathlib import Path
from html.parser import HTMLParser
import json
import os
import subprocess
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.anchors = set()
        self.links = []
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        for key in ("id", "name"):
            if attributes.get(key):
                self.anchors.add(attributes[key])
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(attributes[key])

    def handle_data(self, data):
        self.text.append(data)


def test_pydoc_reference_covers_modules_and_public_api():
    script = Path(__file__).resolve().parents[1] / "pydoc_reference.py"
    result = subprocess.run(
        [sys.executable, str(script)], check=True, capture_output=True, text=True,
    )
    references = json.loads(result.stdout)
    for module in ("dmm", "dmm.config", "dmm.datasets", "dmm.execution",
                   "dmm.benchmark", "dmm.analysis", "dmm.cli",
                   "dmm.algorithms", "dmm.algorithms.cannon", "dmm.matmul", "dmm.trace", "dmm.bsp", "dmm._native"):
        assert f"{module}.html" in references
    pages = {name: Page(html) for name, html in references.items() if name.endswith(".html")}
    cannon = pages["dmm.algorithms.cannon.html"]
    assert "-matmul" in cannon.anchors
    assert "-validate_size" in cannon.anchors
    assert "-matmul_traced" in cannon.anchors
    for name in ("Trace", "SuperstepRecord", "MessageRecord", "Counts", "-count", "-bsp_cost"):
        assert name in pages["dmm._native.html"].anchors, name
    assert "PreparedMatmul" in pages["dmm._native.html"].anchors
    assert "-prepare" in pages["dmm.matmul.html"].anchors
    assert "ThreadExecutor" in pages["dmm._native.html"].anchors
    assert "SerialExecutor" in pages["dmm._native.html"].anchors
    assert "Executor" in pages["dmm._native.html"].anchors
    assert "python/dmm/algorithms/cannon.py" in "".join(cannon.text)
    assert str(script.parents[1]) not in result.stdout
    assert "_compute_tile(" not in result.stdout
    assert references["_pydoc.css"]
    for name, page in pages.items():
        assert references[name].startswith("<!DOCTYPE html>")
        assert 'name="viewport"' in references[name]
        for target in page.links:
            url = urlsplit(target)
            assert url.scheme != "file", target
            if url.scheme or url.netloc or url.path in (".", "./"):
                continue
            destination = url.path or name
            assert destination in references, (name, target)
            if url.fragment:
                assert unquote(url.fragment) in pages[destination].anchors, (name, target)


def test_pydoc_discovers_new_algorithms_and_refreshes_docstrings(tmp_path):
    package = tmp_path / "python/dmm/algorithms"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text('"""Example package."""\n')
    (package / "__init__.py").write_text("")
    algorithm = package / "example.py"
    algorithm.write_text('def matmul(A, B, executor):\n    """Initial documentation."""\n')
    script = Path(__file__).resolve().parents[1] / "pydoc_reference.py"

    def generate():
        result = subprocess.run(
            [sys.executable, "-S", str(script)], check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(tmp_path / "python")},
        )
        return json.loads(result.stdout)

    reference = " ".join("".join(Page(generate()["dmm.algorithms.example.html"]).text).split())
    assert "Initial documentation." in reference
    algorithm.write_text('def matmul(A, B, executor):\n    """Updated API documentation."""\n')
    refreshed = " ".join("".join(Page(generate()["dmm.algorithms.example.html"]).text).split())
    assert "Updated API documentation." in refreshed
    assert "Initial documentation." not in refreshed
