# CLAUDE.md

## Build

```bash
cd /home/robin/mrhash && rm -rf build && pixi run pip uninstall mrhash -y && pixi run pip install -e . --no-build-isolation
```

## Run

Enter the virtual environment named `mrhash` first:

```bash
pixi shell
```
