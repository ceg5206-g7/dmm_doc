"""Generate the API reference with the project's Python environment."""

from pathlib import Path
import json
import subprocess


def on_files(files, config, **kwargs):
    from mkdocs.structure.files import File

    project = Path(config.config_file_path).resolve().parent.parent
    if not (project / "python/dmm/__init__.py").is_file():
        raise RuntimeError("Build the documentation inside the main dmm checkout; pydoc needs its source code.")
    result = subprocess.run(
        ["uv", "run", "--project", str(project), "python", str(Path(__file__).resolve())],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"pydoc generation failed:\n{result.stderr}")
    references = json.loads(result.stdout)
    index = files.get_file_from_path("api/index.md")
    index.content_string += "\n## Modules\n\n| Module |\n| --- |\n"
    for filename, content in references.items():
        files.append(File.generated(config, f"api/{filename}", content=content))
        if filename.endswith(".html"):
            name = filename.removesuffix(".html")
            index.content_string += f"| [`{name}`]({filename}) |\n"
    # shortcut: native HTML bypasses MkDocs search; add indexing if member search is needed.
    return files


if __name__ == "__main__":
    from importlib import import_module
    from importlib.util import find_spec
    from importlib.resources import files
    import pkgutil
    import pydoc

    import dmm

    project = Path(dmm.__file__).resolve().parents[2]
    modules = ["dmm", *sorted(info.name for info in pkgutil.walk_packages(dmm.__path__, "dmm."))]
    # Editable installs can keep the extension outside the source package directory.
    if "dmm._native" not in modules and find_spec("dmm._native") is not None:
        modules.append("dmm._native")

    class ProjectHTMLDoc(pydoc.HTMLDoc):
        def filelink(self, url, path):
            return self.escape(Path(path).relative_to(project).as_posix())

        def modulelink(self, module):
            if module.__name__ not in modules:
                return self.escape(module.__name__)
            return super().modulelink(module)

        def classlink(self, cls, modname):
            if cls.__module__ not in modules:
                return self.escape(pydoc.classname(cls, modname))
            return super().classlink(cls, modname)

        def namelink(self, name, *mappings):
            for mapping in mappings:
                target = mapping.get(name)
                if target and (target.startswith("#") or target.split(".html", 1)[0] in modules):
                    return super().namelink(name, {name: target})
            return name

    renderer = ProjectHTMLDoc()
    references = {}
    for name in modules:
        module = import_module(name)
        content = renderer.docmodule(module).replace('<a href=".">index</a>', '<a href=".">API index</a>')
        reference = renderer.page(name, content)
        references[f"{name}.html"] = reference.replace(
            "</head>", '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<link rel="stylesheet" href="_pydoc.css">\n</head>'
        )
    references["_pydoc.css"] = files("pydoc_data").joinpath("_pydoc.css").read_text(encoding="utf-8")
    print(json.dumps(references))
