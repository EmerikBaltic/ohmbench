# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller --onedir build specification for Ohmbench (Windows)."""

from pathlib import Path

ROOT_DIR = Path(SPECPATH).resolve().parent
SRC_DIR = ROOT_DIR / "src"
PKG_DIR = SRC_DIR / "circuitsim"

block_cipher = None

a = Analysis(
    [str(PKG_DIR / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[
        (str(PKG_DIR / "resources"), "circuitsim/resources"),
        (str(PKG_DIR / "templates"), "circuitsim/templates"),
    ],
    hiddenimports=[
        "scipy.linalg",
        "scipy.special",
        "circuitsim.components.oscilloscope",
        "circuitsim.scope.dso_engine",
        "circuitsim.app.dso_panel",
        "circuitsim.io.spice_export",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Ohmbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Ohmbench",
)
