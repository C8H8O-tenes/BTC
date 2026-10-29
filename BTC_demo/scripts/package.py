"""Package code, models, resources and example data into BTC_demo_full.zip."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
dest = root / 'dist/BTC_demo_full.zip'
dest.parent.mkdir(exist_ok=True)
with ZipFile(dest, 'w', compression=ZIP_DEFLATED, compresslevel=4) as archive:
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(p in {'outputs','graphs','predictions','dist','relocated','__pycache__','.ipynb_checkpoints','.git'} for p in rel.parts):
            continue
        if path.is_file() and path.suffix not in {'.pyc','.log'}:
            archive.write(path, Path('BTC_demo') / rel)
print(dest)
