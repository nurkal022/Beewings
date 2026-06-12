# PyInstaller spec for BeeWings (Windows portable onedir build).
# Build on a Windows machine:  pyinstaller packaging\beewings.spec
#
# Heavy scientific libs (torch, skimage, scipy, matplotlib, cv2, albumentations)
# ship data files and dynamically import submodules that PyInstaller's static
# analysis misses, so we pull them in wholesale with collect_all.

from PyInstaller.utils.hooks import collect_all

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
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

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
    name="beewings",
    console=False,          # GUI app — no console window
    disable_windowed_traceback=False,
    icon=None,              # put an .ico path here if you make one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="BeeWings",        # -> dist/BeeWings/
)
