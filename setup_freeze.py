import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages":
        ["os", "PIL", "colorsys", "random", "math", "random", "argparse"],
    "excludes": ["tkinter"]
    }

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup( name = "PaletteSampler",
        version = "1.0",
        description = "Extract color palette from images",
        options = {"build_exe": build_exe_options},
        executables = [Executable("palette_extractor/paletteSampler.py", base=base)])