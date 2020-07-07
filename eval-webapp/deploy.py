import os
import re
import shutil
import sys
from pathlib import Path


def main():
    deploy_dir = Path(sys.argv[1])
    template_dir = Path("application/templates")
    for file in template_dir.iterdir():
        if file.suffix != ".html":
            continue
        with file.open() as f:
            content = f.read()
        content = re.sub(r"\{% if debug %}(.*?)\{% else %}(.*?)\{% endif %}", r"\2", content, flags=re.DOTALL)
        content = re.sub(r"\{- static_version -}", "deploy", content)
        with (deploy_dir / file.name).open("w") as f:
            f.write(content)
    os.makedirs(deploy_dir / "static/data")
    os.makedirs(deploy_dir / "static/favicon")
    shutil.copy2("application/static/favicon/favicon.png", deploy_dir / "static/favicon/favicon.png")
    shutil.copy2("application/static/app.css", deploy_dir / "static")
    shutil.copy2("application/static/app.js", deploy_dir / "static")


if __name__ == '__main__':
    main()
