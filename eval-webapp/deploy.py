import re
import shutil
from pathlib import Path

import flutes
from argtyped import Arguments


class Args(Arguments):
    deploy_dir: str


def main():
    args = Args()
    deploy_dir = Path(args.deploy_dir)
    template_dir = Path("application/templates")
    for file in template_dir.iterdir():
        with file.open() as f:
            content = f.read()
        content = re.sub(r"{% if debug %}(.*?){% else %}(.*?){% endif %}", r"\2", content)
        content = re.sub(r"{- static_version -}", "deploy", content)
        with (deploy_dir / file.name).open("w") as f:
            f.write(content)
    shutil.rmtree(deploy_dir / "static")
    shutil.copy2("application/static/app.css", deploy_dir / "static")
    shutil.copy2("application/static/app.js", deploy_dir / "static")
    shutil.copytree("application/static/data", deploy_dir / "static" / "data")
    shutil.copytree("application/static/favicon", deploy_dir / "static" / "favicon")


if __name__ == '__main__':
    main()
