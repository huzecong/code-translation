import json
from pathlib import Path
from typing import Any, Dict, IO

import yaml

__all__ = [
    "merge_dict",
    "load_yaml",
]


def merge_dict(template: Dict[str, Any], override: Dict[str, Any]) -> None:
    r"""Merge the ``override`` structure into ``template``, in-place."""
    if len(override) > 0 and not isinstance(template, dict):
        raise ValueError("Template is not a mapping")
    for key, value in override.items():
        if key in template and isinstance(value, dict):
            merge_dict(template[key], value)
        else:
            template[key] = value


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream) -> None:
        try:
            self._root_dir = Path(stream.name).parent
        except AttributeError:
            self._root_dir = Path.cwd()

        super().__init__(stream)

    def _load_file(self, path: str) -> Dict[str, Any]:
        filename = (self._root_dir / path).absolute()
        with filename.open() as f:
            if filename.suffix in ('.yaml', '.yml'):
                return yaml.load(f, Loader)
            elif filename.suffix in ('.json',):
                return json.load(f)
            else:
                return f.read()

    def construct_include(self, node: yaml.Node):
        """Include file referenced at node."""
        return self._load_file(self.construct_scalar(node))

    def construct_template(self, node: yaml.Node):
        item = self.construct_mapping(node, deep=True)
        template = self._load_file(item["path"])
        if not isinstance(template, dict):
            raise ValueError("Root element in file loaded from '!template' tag must be a mapping")
        overrides = item.get("overrides", [])
        if not isinstance(overrides, list):
            overrides = [overrides]
        for override in overrides:
            merge_dict(template, override)
        return template


Loader.add_constructor('!include', Loader.construct_include)
Loader.add_constructor('!template', Loader.construct_template)


def load_yaml(f: IO[str]) -> Dict[str, Any]:
    return yaml.load(f, Loader)
