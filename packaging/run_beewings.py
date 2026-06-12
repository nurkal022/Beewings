"""PyInstaller entry script for BeeWings.

A plain module path can't be a PyInstaller entry point, so we wrap the
console-script target (beewings.app.app:main) in a real .py file.
"""
from beewings.app.app import main

if __name__ == "__main__":
    main()
