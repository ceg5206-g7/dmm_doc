# dmm Documentation

Documentation for [ceg5206-g7/dmm](https://github.com/ceg5206-g7/dmm), maintained
in [dmm_doc](https://github.com/ceg5206-g7/dmm_doc) and checked out as `doc/`
inside the main project. The MkDocs layout, pydoc hook, Mermaid and MathJax
support follow [matmul_doc](https://github.com/ceg5201-group5/matmul_doc).

## Build and preview

Use Python 3.12, uv, CMake 3.21+ and a C++20 compiler. From the main project:

```sh
git submodule update --init --recursive
uv sync --locked
cd doc
uvx --with-requirements requirements.txt mkdocs build --strict
uvx --with-requirements requirements.txt mkdocs serve --strict
```

Open <http://127.0.0.1:8000/> for the preview. Stop it with Ctrl+C.
For a new checkout, clone the main project with `--recurse-submodules`.
After C++ changes, run `uv sync --locked --reinstall-package dmm` before building
the reference. The hook imports the installed extension and current Python
sources using the main project's environment. A standalone documentation clone
must be placed at `dmm/doc/` before running these commands.

MkDocs and its extensions are pinned in `requirements.txt` and run in an isolated
tool environment. The strict build checks document links and anchors. Generated
files go to ignored `site/`; keep authored content under `docs/`. The API hook
generates pydoc HTML and CSS in memory and does not modify source Markdown.
New Python modules are discovered automatically. Keep imports free of benchmark
runs and worker creation.

Run the generator tests from the main project:

```sh
uv run --locked pytest doc/tests -q
```

## Structure

- `docs/`: usage, architecture, algorithms, measurements and contribution guides.
- `docs/api/`: Python reference introduction and the C++ interface reference.
- `pydoc_reference.py`: build hook generating Python module pages.
- `tests/`: API generator checks.
- `mkdocs.yml`: navigation, watched paths and Markdown extensions.

Mermaid and MathJax use pinned CDN URLs from the original documentation framework.
Rendering diagrams and equations requires JavaScript and access to jsDelivr.
Native pydoc pages support browser Find; MkDocs search indexes their module
directory, rather than every generated member.

## Updates and hosting

Edit `docs/` and update navigation when adding a page. Run the generator tests
and strict build before committing. Commit and push this repository first,
then commit its updated `doc/` submodule pointer in `dmm`.

GitHub Actions builds and deploys the MkDocs site on pushes to `main`.
The workflow checks out `dmm` alongside the documentation to build the
native extension and generate the API reference. [Published documentation](https://ceg5206-g7.github.io/dmm_doc/)
