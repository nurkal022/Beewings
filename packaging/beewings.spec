# PyInstaller spec for BeeWings (Windows portable onedir build).
# Build on a Windows machine:  pyinstaller packaging\beewings.spec
#
# Heavy scientific libs (torch, skimage, scipy, matplotlib, cv2, albumentations)
# ship data files and dynamically import submodules that PyInstaller's static
# analysis misses, so we pull them in wholesale with collect_all.

import os
from PyInstaller.utils.hooks import collect_all

HERE = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.dirname(HERE)

datas, binaries, hiddenimports = [], [], []
for pkg in (
    "torch",
    "torchvision",
    "skimage",
    "scipy",
    "matplotlib",
    "albumentations",
    "cv2",
    "PIL",
    "pydantic",
    "openpyxl",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Bundle the ML models next to the exe (dist/BeeWings/checkpoints/...) when
# present on the build machine; resolve_checkpoint() finds them at runtime.
_ckpt_dir = os.path.join(ROOT, "checkpoints")
if os.path.isdir(_ckpt_dir):
    for _f in os.listdir(_ckpt_dir):
        if _f.endswith(".pt"):
            datas.append((os.path.join(_ckpt_dir, _f), "checkpoints"))

_ico = os.path.join(ROOT, "assets", "beewings.ico")
_icon = _ico if os.path.isfile(_ico) else None

a = Analysis(
    ["run_beewings.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BeeWings",
    console=False,          # GUI app — no console window
    disable_windowed_traceback=False,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="BeeWings",        # -> dist/BeeWings/
)
