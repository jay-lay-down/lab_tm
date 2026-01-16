# lab_tm

## Build (PyInstaller)

When packaging, include `openpyxl` as a hidden import so Excel export/import works in the EXE bundle.

```bash
pyinstaller --onefile --hidden-import openpyxl app.py
```
