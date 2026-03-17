# CLAUDE.md

用中文回答

## Build

```bash
cd /home/robin/mrhash && pixi run pip uninstall mrhash -y && rm -rf build && pixi run pip install -e . --no-build-isolation 
```

## Run

Enter the virtual environment named `mrhash` first:

```bash
pixi shell
```
