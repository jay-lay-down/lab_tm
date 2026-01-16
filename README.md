# lab_tm

## Windows build command (SentiWordNet 포함)

```
cd /d C:\Users\70089004\text_file && ^
py -3.12 -m venv .venv_build && ^
call .venv_build\Scripts\activate.bat && ^
python -m pip install -U pip setuptools wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org && ^
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ^
  pyinstaller ^
  pyqt5 ^
  matplotlib ^
  networkx ^
  pandas ^
  requests ^
  openpyxl ^
  pillow ^
  wordcloud ^
  kiwipiepy ^
  kiwipiepy-model ^
  nltk && ^
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "TextMiningTool" ^
  --add-data "Pretendard-Medium.otf;." ^
  --add-data "SentiWord_Dict.txt;." ^
  --add-data "nltk_data;nltk_data" ^
  --hidden-import "matplotlib.backends.backend_qt5agg" ^
  --hidden-import "wordcloud.wordcloud" ^
  --hidden-import "kiwipiepy_model" ^
  --collect-all "kiwipiepy" ^
  --collect-all "kiwipiepy_model" ^
  app.py
```
