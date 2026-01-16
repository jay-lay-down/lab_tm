import importlib.util
import itertools
import json
import os
import random
import re
import sys
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List

_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
_argos_dir = _base / "argos_packages"
os.environ["ARGOS_TRANSLATE_PACKAGES_DIR"] = str(_argos_dir)
os.environ["ARGOS_PACKAGE_DIR"] = str(_argos_dir)

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib import font_manager as fm
import networkx as nx
import numpy as np
import pandas as pd
import requests
from kiwipiepy import Kiwi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from openpyxl.drawing.image import Image as XLImage
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QRadioButton,
    QSplitter,
    QSpinBox,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from wordcloud import WordCloud

if getattr(sys, "_MEIPASS", None):
    bundled_nltk_path = Path(sys._MEIPASS) / "nltk_data"
    if bundled_nltk_path.exists() and str(bundled_nltk_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(bundled_nltk_path))

KNU_DICT_URL = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json"
DEFAULT_RESOURCE_DIR = Path(r"C:\Users\70089004\text_file")
DEFAULT_FONT_NAME = "Pretendard-Medium.otf"
DEFAULT_SENTI_NAME = "SentiWord_Dict.txt"
DEFAULT_EN_SENTI_NAME = "SentiWord_EN.txt"
DEFAULT_NETWORK_FONT_NAME = "malgun.ttf"
DEFAULT_NLTK_DATA_DIR = "nltk_data"
DEFAULT_ARGOS_MODELS_DIR = "argos_models"
PERIOD_ALL_LABEL = "전체 기간"
FALLBACK_FONT_NAMES = [
    "Pretendard",
    "Malgun Gothic",
    "AppleGothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]
SENTIMENT_COLORS = {
    2: "#1f4e99",
    1: "#4a86e8",
    0: "#9e9e9e",
    -1: "#e53935",
    -2: "#b71c1c",
}
EMOTICON_PATTERN = re.compile(r"(ㅠㅠ|ㅜㅜ|ㅎㅎ|ㅋㅋ|ㅋ{2,}|ㅎ{2,})")
PROFANITY_PATTERN = re.compile(r"(ㅅㅂ|시발|씨발|ㅆㅂ|존나|ㅈㄴ|개)")
POSITIVE_PATTERN = re.compile(r"(좋다|좋아|최고|추천|만족|굿|대박|good|굿)")
NEGATION_TOKENS_KO = {"안", "못", "별로", "전혀", "아니", "없", "않"}
NEGATION_TOKENS_EN = {
    "not",
    "no",
    "never",
    "none",
    "n't",
    "cannot",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "shouldn't",
    "couldn't",
    "wouldn't",
    "without",
    "hardly",
    "barely",
}
CONTRAST_TOKENS = ["하지만", "근데", "그런데", "그러나", "but", "however", "though", "although", "yet"]
CONTRACTIONS_MAP = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
    "that's": "that is",
}
DASH_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "―": "-",
        "‐": "-",
    }
)
QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "«": '"',
        "»": '"',
    }
)
EN_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DEFAULT_COUNTRY_LANG_MAP = {
    "KR": "ko",
    "CN": "zh",
    "JP": "ja",
    "IN": "hi",
    "FR": "fr",
    "VN": "vi",
    "TH": "th",
    "ES": "es",
    "IT": "it",
    "SA": "ar",
    "AE": "ar",
    "EG": "ar",
    "US": "en",
    "GB": "en",
}
SUPPORTED_LANG_CODES = {"ko", "en", "zh", "ja", "hi", "fr", "vi", "th", "es", "it", "ar"}


def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent))
    return str(Path(base) / rel_path)


def candidate_resource_paths(filename: str) -> List[Path]:
    return [
        Path(resource_path(filename)),
        DEFAULT_RESOURCE_DIR / filename,
        Path(__file__).resolve().parent / filename,
    ]


def first_existing_path(filename: str) -> Path | None:
    for path in candidate_resource_paths(filename):
        if path.exists():
            return path
    return None


def resolve_font_path() -> str | None:
    font_path = first_existing_path(DEFAULT_FONT_NAME)
    if font_path:
        return str(font_path)
    for font in fm.fontManager.ttflist:
        if font.name in FALLBACK_FONT_NAMES:
            return font.fname
    return None


def resolve_network_font_path() -> str | None:
    network_font_path = first_existing_path(DEFAULT_NETWORK_FONT_NAME)
    if network_font_path:
        return str(network_font_path)
    return resolve_font_path()


def resolve_font_name(font_path: str | None) -> str | None:
    if not font_path:
        return None
    fm.fontManager.addfont(font_path)
    return fm.FontProperties(fname=font_path).get_name()


def configure_matplotlib_font(font_path: str | None):
    if not font_path:
        return
    font_name = resolve_font_name(font_path)
    if not font_name:
        return
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def parse_score_value(raw):
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def parse_sentiment_entries(entries):
    senti_dict = {}
    for entry in entries:
        if isinstance(entry, dict):
            root = entry.get("word_root", entry.get("word"))
            score = parse_score_value(entry.get("polarity", 0))
            if root and score is not None:
                senti_dict[root] = score
            continue
        if isinstance(entry, str):
            parts = re.split(r"[\t,]", entry.strip())
            if len(parts) >= 2:
                root = parts[0].strip()
                if root:
                    score = parse_score_value(parts[1].strip())
                    if score is None:
                        continue
                    senti_dict[root] = score
    return senti_dict


def load_knu_dictionary(parent=None):
    for path in candidate_resource_paths(DEFAULT_SENTI_NAME):
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = content.splitlines()
            senti_dict = parse_sentiment_entries(data)
            if senti_dict:
                return senti_dict

    selected_path, _ = QFileDialog.getOpenFileName(
        parent,
        "감성사전 파일 선택",
        str(DEFAULT_RESOURCE_DIR),
        "Dictionary Files (*.txt *.json);;All Files (*)",
    )
    if selected_path:
        with open(selected_path, "r", encoding="utf-8") as file:
            content = file.read()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = content.splitlines()
        senti_dict = parse_sentiment_entries(data)
        if senti_dict:
            return senti_dict

    try:
        response = requests.get(KNU_DICT_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return parse_sentiment_entries(data)
    except requests.RequestException as exc:
        QMessageBox.warning(
            parent,
            "Dictionary Error",
            "온라인 감성사전 로드 실패.\n"
            "내장/로컬 사전을 확인하거나 파일 선택을 이용해주세요.\n"
            f"{exc}",
        )
    return {}


def load_english_dictionary(parent=None):
    if not ensure_nltk_resources(parent):
        return {}
    senti_dict = build_sentiwordnet_lexicon()
    if senti_dict:
        return senti_dict
    QMessageBox.warning(
        parent,
        "Dictionary Error",
        "SentiWordNet 감성사전 로드 실패.\n"
        "NLTK 리소스를 확인해주세요.",
    )
    return {}


def get_nltk_data_paths() -> list[Path]:
    return [
        Path(resource_path(DEFAULT_NLTK_DATA_DIR)),
        DEFAULT_RESOURCE_DIR / DEFAULT_NLTK_DATA_DIR,
        Path(__file__).resolve().parent / DEFAULT_NLTK_DATA_DIR,
    ]


def ensure_nltk_resources(parent=None) -> bool:
    for path in get_nltk_data_paths():
        if str(path) not in nltk.data.path:
            nltk.data.path.append(str(path))
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/sentiwordnet": "sentiwordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    missing = []
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)
    if missing:
        download_dir = None
        for path in get_nltk_data_paths():
            try:
                path.mkdir(parents=True, exist_ok=True)
                download_dir = str(path)
                break
            except OSError:
                continue
        for name in missing:
            nltk.download(name, quiet=True, download_dir=download_dir)
    still_missing = []
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            still_missing.append(name)
    if still_missing:
        QMessageBox.warning(
            parent,
            "NLTK Resource Error",
            "NLTK 리소스를 다운로드하지 못했습니다.\n"
            f"누락 항목: {', '.join(still_missing)}\n"
            f"NLTK 경로: {', '.join(str(p) for p in get_nltk_data_paths())}",
        )
        return False
    return True


@lru_cache(maxsize=1)
def build_sentiwordnet_lexicon():
    senti_dict = {}
    counts = {}
    for senti_synset in swn.all_senti_synsets():
        score = senti_synset.pos_score() - senti_synset.neg_score()
        if score == 0:
            continue
        for lemma in senti_synset.synset.lemma_names():
            token = wordnet.morphy(lemma) or lemma
            token = token.replace("_", " ").lower()
            senti_dict[token] = senti_dict.get(token, 0) + score
            counts[token] = counts.get(token, 0) + 1
    for token, total in senti_dict.items():
        senti_dict[token] = total / counts[token]
    return senti_dict


def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    parts = re.split(r"[.!?。！？\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def parse_brand_dictionary(raw_text: str):
    brand_map = {}
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            continue
        brand, keywords = line.split(":", 1)
        items = [kw.strip() for kw in re.split(r"[|,]", keywords) if kw.strip()]
        if items:
            brand_map[brand.strip()] = items
    return brand_map


ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
SPACE_RE = re.compile(r"\s+")


def normalize_term(value: str) -> str:
    if value is None:
        return ""
    text = ZERO_WIDTH_RE.sub("", str(value)).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    text = SPACE_RE.sub("", text)
    if not re.search(r"[가-힣A-Za-z0-9]", text):
        return ""
    return text


def normalize_column_name(name: str):
    return re.sub(r"[\s_]", "", name.lower())


def safe_strftime(value):
    if isinstance(value, (datetime, pd.Timestamp)) and pd.notnull(value):
        return value.strftime("%Y-%m-%d")
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        parsed = pd.NaT
    if isinstance(parsed, (pd.Timestamp, datetime)) and pd.notnull(parsed):
        return parsed.strftime("%Y-%m-%d")
    return ""


class ChartCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(6, 4))
        super().__init__(fig)
        self.setParent(parent)

    def add_bar_label_note(self, note_text: str):
        if not note_text:
            return
        self.ax.text(
            0.99,
            1.02,
            note_text,
            transform=self.ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    def format_bar_label(self, count: float, percent: float) -> str:
        count_k = count / 1000
        return f"{count_k:.2f}\n{percent:.1f}%"

    def format_count_label(self, count: float) -> str:
        return f"{int(round(count)):,}"

    def format_percent_label(self, percent: float) -> str:
        return f"{percent:.1f}%"

    def annotate_bars(self, bars, counts, percents, label_mode="both"):
        for bar, count, percent in zip(bars, counts, percents):
            if count <= 0:
                continue
            if label_mode == "count":
                label_text = self.format_count_label(count)
            elif label_mode == "percent":
                label_text = self.format_percent_label(percent)
            else:
                label_text = self.format_bar_label(count, percent)
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                label_text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if bar.get_height() > 0 else "#333333",
            )

    def plot_bar(
        self,
        labels,
        values,
        title,
        ylabel,
        count_values=None,
        note_text="",
        show_labels=True,
    ):
        self.ax.clear()
        if labels:
            bars = self.ax.bar(labels, values, color="#4b77be")
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            if count_values is not None and show_labels:
                total = sum(count_values) or 1
                percents = [(count / total) * 100 for count in count_values]
                self.annotate_bars(bars, count_values, percents)
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.add_bar_label_note(note_text)
        self.figure.tight_layout()
        self.draw()

    def plot_line(self, labels, values, title, ylabel):
        self.ax.clear()
        if labels:
            self.ax.plot(labels, values, marker="o", color="#2c3e50")
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.figure.tight_layout()
        self.draw()

    def plot_multi_line(self, labels, series, title, ylabel, colors_map=None):
        self.ax.clear()
        if labels and series:
            for name, values in series:
                color = colors_map.get(name) if colors_map else None
                self.ax.plot(labels, values, marker="o", label=name, color=color)
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.figure.tight_layout()
        self.draw()

    def plot_multi_bar(
        self,
        labels,
        series,
        title,
        ylabel,
        count_series=None,
        note_text="",
        show_labels=True,
    ):
        self.ax.clear()
        if labels and series:
            total = len(series)
            width = 0.8 / total
            x_positions = list(range(len(labels)))
            for idx, (name, values) in enumerate(series):
                offset = (idx - (total - 1) / 2) * width
                positions = [x + offset for x in x_positions]
                bars = self.ax.bar(positions, values, width=width, label=name)
                if count_series is not None and show_labels:
                    counts = count_series[idx]
                    totals = [sum(group) or 1 for group in zip(*count_series)]
                    percents = [(count / total) * 100 for count, total in zip(counts, totals)]
                    self.annotate_bars(bars, counts, percents)
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.add_bar_label_note(note_text)
        self.figure.tight_layout()
        self.draw()

    def plot_stacked_bar(
        self,
        labels,
        series,
        title,
        ylabel,
        count_series=None,
        note_text="",
        show_labels=True,
        label_mode="both",
    ):
        self.ax.clear()
        if labels and series:
            x_positions = list(range(len(labels)))
            bottoms = [0] * len(labels)
            totals = None
            if count_series is not None:
                totals = [sum(group) or 1 for group in zip(*count_series)]
            for idx, (name, values) in enumerate(series):
                bars = self.ax.bar(x_positions, values, bottom=bottoms, label=name)
                if count_series is not None and totals is not None and show_labels:
                    counts = count_series[idx]
                    percents = [(count / total) * 100 for count, total in zip(counts, totals)]
                    for bar, bottom, count, percent in zip(bars, bottoms, counts, percents):
                        if count <= 0:
                            continue
                        if label_mode == "count":
                            label_text = self.format_count_label(count)
                        elif label_mode == "percent":
                            label_text = self.format_percent_label(percent)
                        else:
                            label_text = self.format_bar_label(count, percent)
                        self.ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bottom + bar.get_height() / 2,
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if bar.get_height() > 0 else "#333333",
                        )
                bottoms = [b + v for b, v in zip(bottoms, values)]
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.add_bar_label_note(note_text)
        self.figure.tight_layout()
        self.draw()

    def plot_stacked_bar_with_labels(self, labels, series, title, ylabel, fmt="{:.1f}%"):
        self.ax.clear()
        if labels and series:
            x_positions = list(range(len(labels)))
            bottoms = [0] * len(labels)
            for name, values in series:
                bars = self.ax.bar(x_positions, values, bottom=bottoms, label=name)
                for bar, bottom, value in zip(bars, bottoms, values):
                    if value <= 0:
                        continue
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2,
                        fmt.format(value),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value >= 15 else "#333333",
                    )
                bottoms = [b + v for b, v in zip(bottoms, values)]
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.figure.tight_layout()
        self.draw()


class TextMiningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Mining Tool")
        self.resize(1400, 900)

        self.df_raw = None
        self.df_clean = None
        self.buzz_df = None
        self.brand_map = {}
        self.stopwords = set()
        self.clean_opts = {}
        self.word_freq_df = None
        self.wc_image_path = None
        self.graph_full = None
        self.graph_view = None
        self.network_pos = None
        self.network_drag_node = None
        self.network_seed_nodes = []
        self.network_level_map = {}
        self.network_stopwords = set()
        self.nodes_df = None
        self.edges_df = None
        self.nodes_view_df = None
        self.edges_view_df = None
        self.sentiment_records_df = None
        self.sentiment_summary_df = None
        self.chart_images = {}
        self.monthly_sampling_enabled = False
        self.monthly_sampling_checkboxes = []
        self.manual_token_replacements = {}
        self.manual_token_merges = []
        self.manual_token_exclusions = set()
        self.preprocess_tables = {}
        self.preprocess_stopwords = set()
        self.preprocess_length_filter = 0
        self.last_wc_topn = []
        self.last_wc_topn_value = 0
        self.last_nodes_ranked = []
        self.base_language = "ko"
        self.language_column = None
        self.language_filter_active = False
        self.language_filter_values = set()
        self.country_language_map = DEFAULT_COUNTRY_LANG_MAP.copy()
        self.text_column = "full_text"

        self.senti_dict = None
        self.senti_max_n = 1
        self.senti_dict_en = None
        self.senti_max_n_en = 1
        self.kiwi = Kiwi()
        self.font_path = resolve_font_path()
        self.network_font_path = resolve_network_font_path()
        self.network_font_name = resolve_font_name(self.network_font_path)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_tab_data_load()
        self._build_tab_buzz()
        self._build_tab_text_mining()
        self._build_tab_preprocess()
        self._build_tab_wordcloud()
        self._build_tab_network()
        self._build_tab_sentiment()
        self._build_tab_export()

        self.footer_label = QLabel(
            'Made by jihee.cho (<a href="https://github.com/jay-lay-down">https://github.com/jay-lay-down</a>)'
        )
        self.footer_label.setOpenExternalLinks(True)
        self.footer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.footer_label)

        self.update_gate_state()
        self.apply_primary_button_styles()
        self.sync_sentiment_language_controls()

    def apply_primary_button_styles(self):
        style = "QPushButton { background-color: #1e88e5; color: white; }"
        for button in [
            self.btn_apply_clean,
            self.btn_refresh_buzz,
            self.btn_build_wc,
            self.btn_build_graph,
            self.btn_run_sentiment,
            self.btn_apply_preprocess,
        ]:
            if button:
                button.setStyleSheet(style)

    def _build_tab_data_load(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizes([200, 600])
        main_layout.addWidget(splitter)

        top = QWidget()
        top.setMinimumHeight(260)
        top.setMaximumHeight(360)
        top_layout = QGridLayout(top)

        self.btn_open_excel = QPushButton("엑셀 파일 열기")
        self.btn_open_excel.clicked.connect(self.load_file)
        self.lbl_file_path = QLabel("파일 미선택")
        self.btn_apply_clean = QPushButton("전처리 적용")
        self.btn_apply_clean.clicked.connect(self.apply_cleaning)
        self.lbl_mapping_status = QLabel("컬럼 매핑 상태: -")
        self.lbl_rows = QLabel("원본 0 → 현재 0")

        top_layout.addWidget(self.btn_open_excel, 0, 0)
        top_layout.addWidget(self.lbl_file_path, 0, 1, 1, 3)
        top_layout.addWidget(self.btn_apply_clean, 0, 4)
        top_layout.addWidget(self.lbl_mapping_status, 1, 0, 1, 3)
        top_layout.addWidget(self.lbl_rows, 1, 3, 1, 2)

        self.group_page_type_filter = QGroupBox("Page Type 제외")
        page_layout = QVBoxLayout(self.group_page_type_filter)
        self.list_page_type = QListWidget()
        self.btn_page_select_all = QPushButton("전체선택")
        self.btn_page_clear = QPushButton("전체해제")
        self.btn_page_select_all.clicked.connect(self.select_all_page_types)
        self.btn_page_clear.clicked.connect(self.clear_all_page_types)
        page_btns = QHBoxLayout()
        page_btns.addWidget(self.btn_page_select_all)
        page_btns.addWidget(self.btn_page_clear)
        page_layout.addLayout(page_btns)
        page_layout.addWidget(self.list_page_type)

        self.group_keyword_filter = QGroupBox("Full Text 키워드 제외")
        kw_layout = QVBoxLayout(self.group_keyword_filter)
        self.le_exclude_keywords = QLineEdit()
        self.le_exclude_keywords.setPlaceholderText("쉼표 또는 | 로 구분")
        self.lbl_exclude_hint = QLabel("입력 키워드 포함 텍스트 제외(대소문자 무시)")
        kw_layout.addWidget(self.le_exclude_keywords)
        kw_layout.addWidget(self.lbl_exclude_hint)

        self.group_language_filter = QGroupBox("기본 언어/언어 필터/번역")
        lang_layout = QGridLayout(self.group_language_filter)
        self.rb_base_lang_ko = QRadioButton("Korean (ko)")
        self.rb_base_lang_en = QRadioButton("English (en)")
        self.rb_base_lang_ko.setChecked(True)
        self.rb_base_lang_ko.toggled.connect(self.handle_base_language_change)
        self.rb_base_lang_en.toggled.connect(self.handle_base_language_change)
        self.cb_language_column = QComboBox()
        self.cb_language_column.addItems(["None"])
        self.cb_language_column.currentIndexChanged.connect(self.update_language_value_list)
        self.btn_edit_lang_map = QPushButton("매핑 테이블 편집")
        self.btn_edit_lang_map.clicked.connect(self.edit_language_mapping)
        self.list_language_values = QListWidget()
        self.list_language_values.itemChanged.connect(self.apply_language_filter_preview)
        self.btn_lang_select_all = QPushButton("전체선택")
        self.btn_lang_clear = QPushButton("전체해제")
        self.btn_lang_select_all.clicked.connect(self.select_all_languages)
        self.btn_lang_clear.clicked.connect(self.clear_all_languages)
        self.chk_lang_filter_active = QCheckBox("선택값만 보기(Filter)")
        self.chk_lang_filter_active.stateChanged.connect(self.apply_language_filter_preview)
        self.lbl_lang_values = QLabel("유니크 값: -")
        self.btn_translate_selected = QPushButton("Translate selected languages → English (offline)")
        self.btn_translate_selected.clicked.connect(self.translate_selected_languages)
        self.chk_translate_overwrite = QCheckBox("Full Text 덮어쓰기")

        lang_layout.addWidget(QLabel("Base Language"), 0, 0)
        lang_layout.addWidget(self.rb_base_lang_ko, 0, 1)
        lang_layout.addWidget(self.rb_base_lang_en, 0, 2)
        lang_layout.addWidget(QLabel("Language column"), 1, 0)
        lang_layout.addWidget(self.cb_language_column, 1, 1)
        lang_layout.addWidget(self.btn_edit_lang_map, 1, 2)
        lang_layout.addWidget(self.lbl_lang_values, 2, 0, 1, 3)
        lang_layout.addWidget(self.list_language_values, 3, 0, 2, 3)
        lang_layout.addWidget(self.btn_lang_select_all, 5, 0)
        lang_layout.addWidget(self.btn_lang_clear, 5, 1)
        lang_layout.addWidget(self.chk_lang_filter_active, 5, 2)
        lang_layout.addWidget(self.btn_translate_selected, 6, 0, 1, 2)
        lang_layout.addWidget(self.chk_translate_overwrite, 6, 2)

        top_layout.addWidget(self.group_page_type_filter, 2, 0, 2, 2)
        top_layout.addWidget(self.group_keyword_filter, 2, 2, 2, 3)
        top_layout.addWidget(self.group_language_filter, 4, 0, 2, 5)

        splitter.addWidget(top)

        self.tbl_preview = QTableWidget()
        self.tbl_preview.setColumnCount(3)
        self.tbl_preview.setHorizontalHeaderLabels(["date", "page_type", "full_text"])
        self.tbl_preview.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.tbl_preview)

        self.tabs.addTab(tab, "데이터 로드")

    def _build_tab_buzz(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QWidget()
        top.setFixedHeight(80)
        top_layout = QGridLayout(top)
        self.cb_granularity = QComboBox()
        self.cb_granularity.addItems(["연도", "월", "주", "일"])
        self.cb_buzz_period_unit = self.cb_granularity
        self.cb_buzz_period_value = QComboBox()
        self.cb_buzz_period_unit.currentIndexChanged.connect(
            lambda: self.populate_period_values(
                self.cb_buzz_period_unit, self.cb_buzz_period_value, self.df_clean
            )
        )
        self.chk_split_by_group = QCheckBox("그룹 분리")
        self.cb_group_by = QComboBox()
        self.cb_group_by.addItem("page_type")
        self.cb_group_by.currentIndexChanged.connect(self.update_group_filter_values)
        self.cb_group_filter = QComboBox()
        self.cb_group_filter.addItem("전체")
        self.cb_buzz_metric = QComboBox()
        self.cb_buzz_metric.addItems(["n", "%"])
        self.btn_refresh_buzz = QPushButton("버즈 계산")
        self.btn_refresh_buzz.clicked.connect(self.build_buzz)

        top_layout.addWidget(QLabel("기간 단위"), 0, 0)
        top_layout.addWidget(self.cb_granularity, 0, 1)
        top_layout.addWidget(QLabel("기간 선택"), 0, 2)
        top_layout.addWidget(self.cb_buzz_period_value, 0, 3)
        top_layout.addWidget(self.chk_split_by_group, 0, 4)
        top_layout.addWidget(QLabel("Group by"), 0, 5)
        top_layout.addWidget(self.cb_group_by, 0, 6)
        top_layout.addWidget(QLabel("필터"), 0, 7)
        top_layout.addWidget(self.cb_group_filter, 0, 8)
        top_layout.addWidget(QLabel("지표"), 0, 9)
        top_layout.addWidget(self.cb_buzz_metric, 0, 10)
        top_layout.addWidget(self.btn_refresh_buzz, 0, 11)

        layout.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([650, 350, 350])
        self.buzz_canvas = ChartCanvas()
        self.buzz_canvas.mpl_connect("button_press_event", self.on_buzz_click)
        self.tbl_buzz = QTableWidget()
        self.tbl_buzz.setColumnCount(0)
        self.tbl_buzz.horizontalHeader().setStretchLastSection(True)
        self.buzz_detail_panel = QGroupBox("버즈 상승 탐색")
        buzz_detail_layout = QVBoxLayout(self.buzz_detail_panel)
        self.lbl_buzz_filters = QLabel("기간/채널: -")
        self.txt_buzz_hot = QTextEdit()
        self.txt_buzz_hot.setReadOnly(True)
        self.txt_buzz_top = QTextEdit()
        self.txt_buzz_top.setReadOnly(True)
        self.txt_buzz_voc = QTextEdit()
        self.txt_buzz_voc.setReadOnly(True)
        buzz_detail_layout.addWidget(self.lbl_buzz_filters)
        buzz_detail_layout.addWidget(QLabel("급상승 단어(핫토픽)"))
        buzz_detail_layout.addWidget(self.txt_buzz_hot)
        buzz_detail_layout.addWidget(QLabel("탑 토픽"))
        buzz_detail_layout.addWidget(self.txt_buzz_top)
        buzz_detail_layout.addWidget(QLabel("VOC 예시"))
        buzz_detail_layout.addWidget(self.txt_buzz_voc)

        splitter.addWidget(self.buzz_canvas)
        splitter.addWidget(self.tbl_buzz)
        splitter.addWidget(self.buzz_detail_panel)
        layout.addWidget(splitter)

        self.tabs.addTab(tab, "버즈량 분석")

    def _build_tab_text_mining(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([600, 800])
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_split = QSplitter(Qt.Vertical)
        left_split.setSizes([320, 320])

        topic_group = QGroupBox("핵심 키워드 설정")
        topic_layout = QVBoxLayout(topic_group)
        topic_row = QHBoxLayout()
        self.le_topic_name = QLineEdit()
        self.le_topic_name.setPlaceholderText("핵심 키워드 1")
        topic_row.addWidget(self.le_topic_name)
        self.txt_topic_related = QTextEdit()
        self.txt_topic_related.setPlaceholderText("관련어를 입력하세요 (쉼표/줄바꿈 구분)")
        self.txt_topic_related.setFixedHeight(90)
        self.btn_apply_brand = QPushButton("토픽 추가")
        self.btn_apply_brand.clicked.connect(self.apply_brand_dict)
        self.btn_save_topics = QPushButton("토픽 저장")
        self.btn_save_topics.clicked.connect(self.save_topic_dictionary)
        self.btn_load_topics = QPushButton("토픽 불러오기")
        self.btn_load_topics.clicked.connect(self.load_topic_dictionary)
        self.list_topics = QListWidget()
        self.list_topics.setMinimumHeight(120)
        topic_buttons = QHBoxLayout()
        self.btn_remove_topic = QPushButton("선택 토픽 삭제")
        self.btn_remove_topic.clicked.connect(self.remove_selected_topic)
        topic_buttons.addWidget(self.btn_save_topics)
        topic_buttons.addWidget(self.btn_load_topics)
        topic_buttons.addStretch()
        topic_buttons.addWidget(self.btn_remove_topic)
        topic_layout.addLayout(topic_row)
        topic_layout.addWidget(self.txt_topic_related)
        topic_layout.addWidget(self.btn_apply_brand)
        topic_layout.addWidget(self.list_topics)
        topic_layout.addLayout(topic_buttons)

        stop_group = QGroupBox("불용어 입력")
        stop_layout = QVBoxLayout(stop_group)
        self.txt_stopwords = QTextEdit()
        self.txt_stopwords.setPlaceholderText("불용어를 줄바꿈/쉼표로 입력")
        opts_row = QHBoxLayout()
        self.chk_remove_numbers = QCheckBox("숫자 제거")
        self.chk_remove_symbols = QCheckBox("특수문자 제거")
        self.chk_remove_single = QCheckBox("길이1 제거")
        self.chk_korean_only = QCheckBox("한글만")
        self.chk_english_only = QCheckBox("영문만")
        opts_row.addWidget(self.chk_remove_numbers)
        opts_row.addWidget(self.chk_remove_symbols)
        opts_row.addWidget(self.chk_remove_single)
        opts_row.addWidget(self.chk_korean_only)
        opts_row.addWidget(self.chk_english_only)
        self.chk_monthly_sample_text = self.create_monthly_sampling_checkbox()
        self.btn_apply_stopwords = QPushButton("불용어/옵션 적용")
        self.btn_apply_stopwords.clicked.connect(self.apply_stopwords)
        stop_layout.addWidget(self.txt_stopwords)
        stop_layout.addLayout(opts_row)
        stop_layout.addWidget(self.chk_monthly_sample_text)
        stop_layout.addWidget(self.btn_apply_stopwords)

        left_split.addWidget(topic_group)
        left_split.addWidget(stop_group)
        left_layout.addWidget(left_split)

        self.tbl_token_sample = QTableWidget()
        self.tbl_token_sample.setColumnCount(2)
        self.tbl_token_sample.setHorizontalHeaderLabels(["token", "count"])
        self.tbl_token_sample.horizontalHeader().setStretchLastSection(True)
        self.tbl_token_sample.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_token_sample.setSelectionMode(QTableWidget.ExtendedSelection)
        self.tbl_token_sample.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_token_sample.customContextMenuRequested.connect(self.show_token_menu)

        splitter.addWidget(left)
        splitter.addWidget(self.tbl_token_sample)

        self.tabs.addTab(tab, "텍스트마이닝 설정")

    def _create_rule_table(self, headers, rows=6):
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setRowCount(rows)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _build_tab_preprocess(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([900, 300])
        layout.addWidget(splitter)

        pre_group = QGroupBox("토큰화 전 규칙")
        pre_layout = QGridLayout(pre_group)
        self.cb_unicode_norm = QComboBox()
        self.cb_unicode_norm.addItems(["사용 안 함", "NFKC", "NFC"])
        self.chk_trim_space = QCheckBox("공백 정리")
        self.chk_unify_quotes = QCheckBox("따옴표/대시 통일")
        self.chk_model_punct = QCheckBox("모델명 구두점 통일 (S-24→S24)")
        self.chk_repeat_reduce = QCheckBox("반복 문자 축약")
        self.chk_lowercase = QCheckBox("영어 소문자화")
        self.cb_contractions = QComboBox()
        self.cb_contractions.addItems(["contractions 미적용", "contractions 적용"])

        self.tbl_join_map = self._create_rule_table(["원문", "치환"], rows=6)
        self.tbl_ko_en_map = self._create_rule_table(["표기", "대표"], rows=6)
        self.tbl_compound_map = self._create_rule_table(["표기", "대표"], rows=6)

        pre_layout.addWidget(QLabel("유니코드 정규화"), 0, 0)
        pre_layout.addWidget(self.cb_unicode_norm, 0, 1)
        pre_layout.addWidget(self.chk_trim_space, 0, 2)
        pre_layout.addWidget(self.chk_unify_quotes, 0, 3)
        pre_layout.addWidget(self.chk_model_punct, 1, 0, 1, 2)
        pre_layout.addWidget(self.chk_repeat_reduce, 1, 2)
        pre_layout.addWidget(self.chk_lowercase, 1, 3)
        pre_layout.addWidget(QLabel("contractions"), 2, 0)
        pre_layout.addWidget(self.cb_contractions, 2, 1)
        pre_layout.addWidget(QLabel("붙여쓰기 치환"), 3, 0, 1, 2)
        pre_layout.addWidget(self.tbl_join_map, 4, 0, 1, 4)
        pre_layout.addWidget(QLabel("한글/영문 표기 통일"), 5, 0, 1, 2)
        pre_layout.addWidget(self.tbl_ko_en_map, 6, 0, 1, 4)
        pre_layout.addWidget(QLabel("복합어 통일"), 7, 0, 1, 2)
        pre_layout.addWidget(self.tbl_compound_map, 8, 0, 1, 4)

        post_group = QGroupBox("토큰화 후 규칙")
        post_layout = QGridLayout(post_group)
        self.tbl_repr_map = self._create_rule_table(["표현", "대표어"], rows=6)
        self.chk_lemmatize = QCheckBox("원형화/표제어 처리")
        self.txt_pre_stopwords = QTextEdit()
        self.txt_pre_stopwords.setPlaceholderText("불용어 입력 (쉼표/줄바꿈)")
        self.sb_min_length = QSpinBox()
        self.sb_min_length.setRange(0, 5)
        self.sb_min_length.setValue(0)
        self.tbl_merge_tokens = self._create_rule_table(["연속 토큰", "병합"], rows=6)
        self.tbl_syn_map = self._create_rule_table(["동의어", "대표"], rows=6)
        self.chk_plural_singular = QCheckBox("복수/단수 통합")
        self.tbl_model_map = self._create_rule_table(["모델명", "대표"], rows=6)

        post_layout.addWidget(QLabel("대표어 매핑"), 0, 0, 1, 2)
        post_layout.addWidget(self.tbl_repr_map, 1, 0, 1, 4)
        post_layout.addWidget(self.chk_lemmatize, 2, 0)
        post_layout.addWidget(QLabel("불용어 제거"), 3, 0)
        post_layout.addWidget(self.txt_pre_stopwords, 3, 1, 1, 3)
        post_layout.addWidget(QLabel("길이 필터(글자 이하 제거)"), 4, 0)
        post_layout.addWidget(self.sb_min_length, 4, 1)
        post_layout.addWidget(QLabel("연속 토큰 병합"), 5, 0, 1, 2)
        post_layout.addWidget(self.tbl_merge_tokens, 6, 0, 1, 4)
        post_layout.addWidget(QLabel("다국어 동의어 통합"), 7, 0, 1, 2)
        post_layout.addWidget(self.tbl_syn_map, 8, 0, 1, 4)
        post_layout.addWidget(self.chk_plural_singular, 9, 0, 1, 2)
        post_layout.addWidget(QLabel("모델명 표기 통일"), 10, 0, 1, 2)
        post_layout.addWidget(self.tbl_model_map, 11, 0, 1, 4)

        pos_group = QGroupBox("품사 선택")
        pos_layout = QHBoxLayout(pos_group)
        self.rb_pos_noun = QRadioButton("명사만")
        self.rb_pos_noun_adj = QRadioButton("명사+형용사")
        self.rb_pos_noun_verb = QRadioButton("명사+동사")
        self.rb_pos_all = QRadioButton("명사+형용사+동사")
        self.rb_pos_noun.setChecked(True)
        pos_layout.addWidget(self.rb_pos_noun)
        pos_layout.addWidget(self.rb_pos_noun_adj)
        pos_layout.addWidget(self.rb_pos_noun_verb)
        pos_layout.addWidget(self.rb_pos_all)

        rules_splitter = QSplitter(Qt.Horizontal)
        rules_splitter.setSizes([520, 520])
        left_rules = QWidget()
        left_rules_layout = QVBoxLayout(left_rules)
        left_rules_layout.addWidget(pre_group)
        right_rules = QWidget()
        right_rules_layout = QVBoxLayout(right_rules)
        right_rules_layout.addWidget(post_group)
        right_rules_layout.addWidget(pos_group)
        rules_splitter.addWidget(left_rules)
        rules_splitter.addWidget(right_rules)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.lbl_preprocess_count = QLabel("토큰 0 / 고유 0")
        self.tbl_preprocess_tokens = QTableWidget()
        self.tbl_preprocess_tokens.setColumnCount(2)
        self.tbl_preprocess_tokens.setHorizontalHeaderLabels(["token", "count"])
        self.tbl_preprocess_tokens.horizontalHeader().setStretchLastSection(True)
        self.tbl_preprocess_tokens.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_preprocess_tokens.setSelectionMode(QTableWidget.ExtendedSelection)
        self.tbl_preprocess_tokens.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_preprocess_tokens.customContextMenuRequested.connect(
            self.show_preprocess_token_menu
        )
        self.btn_save_preprocess_tokens = QPushButton("Save")
        self.btn_save_preprocess_tokens.setMaximumWidth(80)
        self.btn_save_preprocess_tokens.clicked.connect(self.save_preprocess_token_edits)

        right_layout.addWidget(self.lbl_preprocess_count)
        right_layout.addWidget(self.tbl_preprocess_tokens)
        right_layout.addWidget(self.btn_save_preprocess_tokens, alignment=Qt.AlignRight)

        splitter.addWidget(rules_splitter)
        splitter.addWidget(right)

        buttons = QHBoxLayout()
        self.btn_apply_preprocess = QPushButton("적용")
        self.btn_apply_preprocess.clicked.connect(self.apply_preprocess_rules)
        self.btn_reset_preprocess = QPushButton("초기화")
        self.btn_reset_preprocess.clicked.connect(self.reset_preprocess_rules)
        self.btn_save_preprocess_rules = QPushButton("저장")
        self.btn_save_preprocess_rules.clicked.connect(self.save_preprocess_rules)
        buttons.addStretch()
        buttons.addWidget(self.btn_apply_preprocess)
        buttons.addWidget(self.btn_reset_preprocess)
        buttons.addWidget(self.btn_save_preprocess_rules)
        layout.addLayout(buttons)

        self.preprocess_tables = {
            "join": self.tbl_join_map,
            "ko_en": self.tbl_ko_en_map,
            "compound": self.tbl_compound_map,
            "repr": self.tbl_repr_map,
            "merge": self.tbl_merge_tokens,
            "syn": self.tbl_syn_map,
            "model": self.tbl_model_map,
        }

        self.tabs.addTab(tab, "전처리")

    def _build_tab_wordcloud(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QWidget()
        top.setFixedHeight(90)
        top_layout = QGridLayout(top)
        self.cb_wc_period_unit = QComboBox()
        self.cb_wc_period_unit.addItems(["연도", "월"])
        self.cb_wc_period_value = QComboBox()
        self.cb_wc_period_unit.currentIndexChanged.connect(
            lambda: (
                self.populate_period_values(
                    self.cb_wc_period_unit, self.cb_wc_period_value, self.df_clean
                ),
                self.refresh_token_sample(),
            )
        )
        self.cb_wc_period_value.currentIndexChanged.connect(self.refresh_token_sample)
        self.cb_wc_topn = QComboBox()
        self.cb_wc_topn.addItems(["30", "50", "100", "200"])
        self.cb_wc_topn.currentIndexChanged.connect(self.refresh_token_sample)
        self.btn_build_wc = QPushButton("워드클라우드 생성")
        self.btn_build_wc.clicked.connect(self.build_wordcloud)
        self.chk_wc_random_style = QCheckBox("랜덤 팔레트/모양 사용")
        self.chk_monthly_sample_wc = self.create_monthly_sampling_checkbox()
        self.lbl_wc_count = QLabel("토큰 0 / 고유 0")
        top_layout.addWidget(QLabel("기간 단위"), 0, 0)
        top_layout.addWidget(self.cb_wc_period_unit, 0, 1)
        top_layout.addWidget(QLabel("기간 선택"), 0, 2)
        top_layout.addWidget(self.cb_wc_period_value, 0, 3)
        top_layout.addWidget(QLabel("Top N"), 0, 4)
        top_layout.addWidget(self.cb_wc_topn, 0, 5)
        top_layout.addWidget(self.chk_wc_random_style, 0, 6)
        top_layout.addWidget(self.chk_monthly_sample_wc, 0, 7)
        top_layout.addWidget(self.btn_build_wc, 0, 8)
        top_layout.addWidget(self.lbl_wc_count, 0, 9)

        layout.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([700, 500])
        self.lbl_wc_view = QLabel("워드클라우드 미생성")
        self.lbl_wc_view.setAlignment(Qt.AlignCenter)
        self.tbl_wc_topn = QTableWidget()
        self.tbl_wc_topn.setColumnCount(2)
        self.tbl_wc_topn.setHorizontalHeaderLabels(["token", "count"])
        self.tbl_wc_topn.horizontalHeader().setStretchLastSection(True)
        self.tbl_wc_topn.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_wc_topn.setSelectionMode(QTableWidget.ExtendedSelection)
        self.tbl_wc_topn.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_wc_topn.customContextMenuRequested.connect(self.show_wordcloud_menu)
        self.btn_save_wc_table = QPushButton("Save")
        self.btn_save_wc_table.setMaximumWidth(80)
        self.btn_save_wc_table.clicked.connect(self.save_wordcloud_edits)

        splitter.addWidget(self.lbl_wc_view)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.tbl_wc_topn)
        right_layout.addWidget(self.btn_save_wc_table, alignment=Qt.AlignRight)
        splitter.addWidget(right)
        layout.addWidget(splitter)

        self.tabs.addTab(tab, "워드클라우드")

    def _build_tab_network(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizes([150, 180, 470])
        layout.addWidget(splitter)

        top = QWidget()
        top.setFixedHeight(150)
        top_layout = QGridLayout(top)
        self.btn_build_graph = QPushButton("그래프 생성")
        self.btn_build_graph.clicked.connect(self.build_network)
        self.cb_network_period_unit = QComboBox()
        self.cb_network_period_unit.addItems(["연도", "월"])
        self.cb_network_period_value = QComboBox()
        self.cb_network_period_unit.currentIndexChanged.connect(
            lambda: self.populate_period_values(
                self.cb_network_period_unit, self.cb_network_period_value, self.df_clean
            )
        )
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["기본", "고급"])
        self.cb_mode.currentIndexChanged.connect(self.toggle_network_advanced)
        self.sb_min_node_count = QSpinBox()
        self.sb_min_node_count.setRange(1, 100)
        self.sb_min_node_count.setValue(1)
        self.lbl_min_node = QLabel("노드 최소 등장")
        self.sb_min_edge_weight = QSpinBox()
        self.sb_min_edge_weight.setRange(1, 100)
        self.sb_min_edge_weight.setValue(3)
        self.lbl_min_edge = QLabel("엣지 최소 가중치")
        self.sb_max_nodes = QSpinBox()
        self.sb_max_nodes.setRange(10, 300)
        self.sb_max_nodes.setValue(60)
        self.lbl_max_nodes = QLabel("최대 노드 수")

        self.le_node_search = QLineEdit()
        self.le_node_search.setPlaceholderText("노드 검색")
        self.btn_add_seed = QPushButton("노드 선택")
        self.btn_add_seed.clicked.connect(self.select_node_from_search)
        self.cb_hop_depth = QComboBox()
        self.cb_hop_depth.addItems(["1", "2", "3", "4"])
        self.cb_hop_depth.setCurrentIndex(1)
        self.btn_apply_hop = QPushButton("Hop 적용/그리기")
        self.btn_apply_hop.clicked.connect(self.apply_hop)
        self.btn_reset_view = QPushButton("초기화")
        self.btn_reset_view.clicked.connect(self.reset_network_view)

        self.cb_cooc_scope = QComboBox()
        self.cb_cooc_scope.addItems(["문서(로우)", "문장"])
        self.cb_weight_mode = QComboBox()
        self.cb_weight_mode.addItems(["count", "PMI"])
        self.cb_weight_mode.currentIndexChanged.connect(self.handle_pmi_guard)
        self.chk_drag_mode = QCheckBox("노드 위치 편집")
        self.chk_drag_mode.toggled.connect(self.redraw_network)
        self.chk_monthly_sample_network = self.create_monthly_sampling_checkbox()
        self.le_network_stopwords = QLineEdit()
        self.le_network_stopwords.setPlaceholderText("네트워크 전용 불용어 (쉼표/줄바꿈)")
        self.lbl_network_reco = QLabel("데이터 기준 권장값: -")
        self.lbl_advanced_hint = QLabel(
            "PMI는 희귀 단어쌍을 과대평가할 수 있어 min_node_count ≥ 10, "
            "min_edge_weight ≥ 5를 권장합니다. (데이터가 작을수록 필터를 높이세요)"
        )

        top_layout.addWidget(QLabel("기간 단위"), 0, 0)
        top_layout.addWidget(self.cb_network_period_unit, 0, 1)
        top_layout.addWidget(QLabel("기간 선택"), 0, 2)
        top_layout.addWidget(self.cb_network_period_value, 0, 3)
        top_layout.addWidget(self.cb_mode, 0, 4)
        top_layout.addWidget(self.lbl_min_node, 0, 5)
        top_layout.addWidget(self.sb_min_node_count, 0, 6)
        top_layout.addWidget(self.lbl_min_edge, 0, 7)
        top_layout.addWidget(self.sb_min_edge_weight, 0, 8)
        top_layout.addWidget(self.lbl_max_nodes, 0, 9)
        top_layout.addWidget(self.sb_max_nodes, 0, 10)
        top_layout.addWidget(self.le_node_search, 1, 0, 1, 2)
        top_layout.addWidget(self.btn_add_seed, 1, 2)
        top_layout.addWidget(self.cb_hop_depth, 1, 3)
        top_layout.addWidget(self.btn_apply_hop, 1, 4)
        top_layout.addWidget(self.btn_reset_view, 1, 5)
        top_layout.addWidget(self.btn_build_graph, 1, 6)
        top_layout.addWidget(self.cb_cooc_scope, 2, 0, 1, 2)
        top_layout.addWidget(self.cb_weight_mode, 2, 2, 1, 2)
        top_layout.addWidget(self.chk_drag_mode, 2, 4)
        top_layout.addWidget(self.chk_monthly_sample_network, 2, 5)
        top_layout.addWidget(self.le_network_stopwords, 2, 6, 1, 3)
        top_layout.addWidget(self.lbl_network_reco, 2, 9, 1, 2)
        top_layout.addWidget(self.lbl_advanced_hint, 2, 11, 1, 2)

        splitter.addWidget(top)

        self.list_nodes_ranked = QListWidget()
        self.list_nodes_ranked.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_nodes_ranked.itemDoubleClicked.connect(self.toggle_node_selection)
        splitter.addWidget(self.list_nodes_ranked)

        bottom = QSplitter(Qt.Horizontal)
        bottom.setSizes([700, 500])
        self.network_canvas = ChartCanvas()
        self.network_canvas.mpl_connect("button_press_event", self.on_network_press)
        self.network_canvas.mpl_connect("button_release_event", self.on_network_release)
        self.network_canvas.mpl_connect("motion_notify_event", self.on_network_motion)
        self.network_tabs = QTabWidget()
        self.tbl_edges = QTableWidget()
        self.tbl_edges.setColumnCount(3)
        self.tbl_edges.setHorizontalHeaderLabels(["source", "target", "weight"])
        self.tbl_edges.horizontalHeader().setStretchLastSection(True)
        self.tbl_nodes = QTableWidget()
        self.tbl_nodes.setColumnCount(2)
        self.tbl_nodes.setHorizontalHeaderLabels(["node", "degree"])
        self.tbl_nodes.horizontalHeader().setStretchLastSection(True)
        self.network_tabs.addTab(self.tbl_edges, "Edges")
        self.network_tabs.addTab(self.tbl_nodes, "Nodes")
        self.btn_save_network_table = QPushButton("Save")
        self.btn_save_network_table.setMaximumWidth(80)
        self.btn_save_network_table.clicked.connect(self.save_network_edits)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.network_tabs)
        right_layout.addWidget(self.btn_save_network_table, alignment=Qt.AlignRight)
        bottom.addWidget(self.network_canvas)
        bottom.addWidget(right)
        splitter.addWidget(bottom)

        self.tabs.addTab(tab, "네트워크")
        self.toggle_network_advanced()

    def _build_tab_sentiment(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QWidget()
        top.setFixedHeight(210)
        top_layout = QVBoxLayout(top)
        filter_row = QWidget()
        filter_layout = QGridLayout(filter_row)
        self.cb_sent_period_value = QComboBox()
        self.cb_sent_period_value.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sent_mode = QComboBox()
        self.cb_sent_mode.addItems(["전체 감성", "사전별 감성"])
        self.cb_sent_mode.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sent_view = QComboBox()
        self.cb_sent_view.addItems(["연도별", "월별", "주별", "일별"])
        self.cb_sent_view.currentIndexChanged.connect(
            lambda: (self.populate_sentiment_period_values(), self.update_sentiment_view())
        )
        self.cb_sent_metric = QComboBox()
        self.cb_sent_metric.addItems(["count", "%"])
        self.cb_sent_metric.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sentence_split = QComboBox()
        self.cb_sentence_split.addItems(["기본", "강함(쉼표 포함)"])
        self.rb_sent_lang_ko = QRadioButton("한국어")
        self.rb_sent_lang_en = QRadioButton("영어")
        for rb in [self.rb_sent_lang_ko, self.rb_sent_lang_en]:
            rb.toggled.connect(self.handle_sentiment_language_change)
        self.group_sent_lang = QGroupBox("기본 언어(데이터 로드)")
        lang_layout = QHBoxLayout(self.group_sent_lang)
        lang_layout.setContentsMargins(6, 4, 6, 4)
        lang_layout.addWidget(self.rb_sent_lang_ko)
        lang_layout.addWidget(self.rb_sent_lang_en)
        self.cb_brand_filter = QComboBox()
        self.cb_brand_filter.addItem("전체")
        self.cb_brand_filter.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sent_page_type = QComboBox()
        self.cb_sent_page_type.addItem("전체")
        self.cb_sent_page_type.currentIndexChanged.connect(self.update_sentiment_view)
        self.btn_run_sentiment = QPushButton("감성분석 실행")
        self.btn_run_sentiment.clicked.connect(self.run_sentiment)
        self.chk_monthly_sample_sent = self.create_monthly_sampling_checkbox()
        self.lbl_sent_period_range = QLabel("")

        filter_layout.addWidget(QLabel("기간 선택"), 0, 0)
        filter_layout.addWidget(self.cb_sent_period_value, 0, 1)
        filter_layout.addWidget(QLabel("모드"), 0, 2)
        filter_layout.addWidget(self.cb_sent_mode, 0, 3)
        filter_layout.addWidget(QLabel("보기"), 0, 4)
        filter_layout.addWidget(self.cb_sent_view, 0, 5)
        filter_layout.addWidget(QLabel("지표"), 0, 6)
        filter_layout.addWidget(self.cb_sent_metric, 0, 7)
        filter_layout.addWidget(self.chk_monthly_sample_sent, 0, 8)
        filter_layout.addWidget(self.group_sent_lang, 0, 9, 1, 2)
        filter_layout.addWidget(QLabel("문장 분리"), 1, 0)
        filter_layout.addWidget(self.cb_sentence_split, 1, 1)
        filter_layout.addWidget(self.btn_run_sentiment, 1, 2)
        filter_layout.addWidget(QLabel("토픽"), 1, 3)
        filter_layout.addWidget(self.cb_brand_filter, 1, 4, 1, 2)
        filter_layout.addWidget(QLabel("page_type"), 1, 6)
        filter_layout.addWidget(self.cb_sent_page_type, 1, 7)
        filter_layout.addWidget(self.lbl_sent_period_range, 1, 8, 1, 2)
        top_layout.addWidget(filter_row)

        rules_group = QGroupBox("룰 기반 보정")
        rules_layout = QGridLayout(rules_group)
        self.chk_rule_master = QCheckBox("룰 기반 보정 사용(마스터)")
        self.chk_rule_master.setChecked(True)
        self.chk_rule_neg_scope = QCheckBox("부정어 스코프")
        self.chk_rule_contrast = QCheckBox("대조 접속어")
        self.chk_rule_emoticon = QCheckBox("이모티콘+긍정 보정")
        self.chk_rule_profanity = QCheckBox("비속어+긍정 강화")
        self.chk_rule_prev_negative = QCheckBox("이전 문장 부정 전파(약)")
        for checkbox in [
            self.chk_rule_neg_scope,
            self.chk_rule_contrast,
            self.chk_rule_emoticon,
            self.chk_rule_profanity,
            self.chk_rule_prev_negative,
        ]:
            checkbox.setChecked(True)
        rules_layout.addWidget(self.chk_rule_master, 0, 0, 1, 2)
        rules_layout.addWidget(self.chk_rule_neg_scope, 1, 0)
        rules_layout.addWidget(self.chk_rule_contrast, 1, 1)
        rules_layout.addWidget(self.chk_rule_emoticon, 1, 2)
        rules_layout.addWidget(self.chk_rule_profanity, 1, 3)
        rules_layout.addWidget(self.chk_rule_prev_negative, 1, 4)
        top_layout.addWidget(rules_group)

        layout.addWidget(top)

        content_split = QSplitter(Qt.Vertical)
        content_split.setSizes([520, 200])

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([720, 480])
        self.tbl_sent_records = QTableWidget()
        self.tbl_sent_records.setColumnCount(6)
        self.tbl_sent_records.setHorizontalHeaderLabels([
            "date",
            "page_type",
            "sentence",
            "score",
            "label",
            "topic",
        ])
        self.tbl_sent_records.horizontalHeader().setStretchLastSection(True)
        self.tbl_sent_records.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_sent_records.setSelectionMode(QTableWidget.ExtendedSelection)
        self.tbl_sent_records.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_sent_records.customContextMenuRequested.connect(self.show_sentiment_menu)
        self.sent_chart_tabs = QTabWidget()
        self.sent_canvas = ChartCanvas()
        self.sent_topic_charts_container = QWidget()
        self.sent_topic_charts_layout = QVBoxLayout(self.sent_topic_charts_container)
        self.sent_topic_charts_layout.setContentsMargins(8, 8, 8, 8)
        self.sent_topic_charts_layout.setSpacing(12)
        self.sent_topic_scroll = QScrollArea()
        self.sent_topic_scroll.setWidgetResizable(True)
        self.sent_topic_scroll.setWidget(self.sent_topic_charts_container)
        self.sent_topic_canvas = ChartCanvas()
        self.sent_topic_charts_layout.addWidget(self.sent_topic_canvas)
        self.sent_trend_canvas = ChartCanvas()
        self.sent_chart_tabs.addTab(self.sent_canvas, "감성 분포")
        self.sent_chart_tabs.addTab(self.sent_topic_scroll, "사전별 감성 스택")
        self.sent_chart_tabs.addTab(self.sent_trend_canvas, "감성 추이")
        splitter.addWidget(self.tbl_sent_records)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.sent_chart_tabs)
        self.btn_save_sentiment_table = QPushButton("Save")
        self.btn_save_sentiment_table.setMaximumWidth(80)
        self.btn_save_sentiment_table.clicked.connect(self.save_sentiment_edits)
        right_layout.addWidget(self.btn_save_sentiment_table, alignment=Qt.AlignRight)
        splitter.addWidget(right)

        voc_group = QGroupBox("VoC 요약")
        voc_layout = QVBoxLayout(voc_group)
        self.txt_voc = QTextEdit()
        self.txt_voc.setReadOnly(True)
        voc_layout.addWidget(self.txt_voc)

        content_split.addWidget(splitter)
        content_split.addWidget(voc_group)
        layout.addWidget(content_split)

        self.tabs.addTab(tab, "감성분석")

    def _build_tab_export(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([520, 680])
        layout.addWidget(splitter)

        self.list_export_items = QListWidget()
        for label in [
            "clean_data",
            "buzz_summary",
            "word_freq_topN",
            "network_nodes",
            "network_edges",
            "sentiment_records",
            "sentiment_summary",
            "charts",
        ]:
            item = QListWidgetItem(label)
            item.setCheckState(Qt.Checked)
            self.list_export_items.addItem(item)

        right = QWidget()
        right_layout = QGridLayout(right)
        self.lbl_out_dir = QLabel("저장 폴더 미선택")
        self.btn_choose_dir = QPushButton("폴더 선택")
        self.btn_choose_dir.clicked.connect(self.choose_output_dir)
        self.le_filename = QLineEdit("text_mining_result.xlsx")
        self.btn_export_excel = QPushButton("엑셀 저장")
        self.btn_export_excel.clicked.connect(self.export_excel)
        self.txt_export_log = QTextEdit()
        self.txt_export_log.setReadOnly(True)

        right_layout.addWidget(self.lbl_out_dir, 0, 0)
        right_layout.addWidget(self.btn_choose_dir, 0, 1)
        right_layout.addWidget(self.le_filename, 1, 0)
        right_layout.addWidget(self.btn_export_excel, 1, 1)
        right_layout.addWidget(self.txt_export_log, 2, 0, 1, 2)

        splitter.addWidget(self.list_export_items)
        splitter.addWidget(right)

        self.tabs.addTab(tab, "저장")

    def update_gate_state(self):
        enabled = self.df_clean is not None
        for idx in range(1, 8):
            self.tabs.setTabEnabled(idx, enabled)
        if not enabled:
            self.statusBar().showMessage("데이터 로드 후 전처리를 적용해주세요.")
        self.refresh_period_filters()

    def populate_period_values(self, unit_combo: QComboBox, value_combo: QComboBox, df: pd.DataFrame | None):
        value_combo.blockSignals(True)
        value_combo.clear()
        value_combo.addItem(PERIOD_ALL_LABEL)
        if df is None or df.empty or "date" not in df.columns:
            value_combo.blockSignals(False)
            return
        dates = df["date"].dropna()
        if dates.empty:
            value_combo.blockSignals(False)
            return
        unit = unit_combo.currentText()
        if unit == "연도":
            values = sorted({str(val.year) for val in dates})
        elif unit == "월":
            values = sorted({val.strftime("%Y-%m") for val in dates})
        elif unit == "주":
            values = sorted({val.to_period("W").start_time.strftime("%Y-%m-%d") for val in dates})
        else:
            values = sorted({val.strftime("%Y-%m-%d") for val in dates})
        for value in values:
            value_combo.addItem(value)
        value_combo.blockSignals(False)

    def refresh_period_filters(self):
        df = self.df_clean
        for unit_combo, value_combo in [
            (self.cb_buzz_period_unit, self.cb_buzz_period_value),
            (self.cb_wc_period_unit, self.cb_wc_period_value),
            (self.cb_network_period_unit, self.cb_network_period_value),
        ]:
            self.populate_period_values(unit_combo, value_combo, df)
        self.populate_sentiment_period_values()

    def populate_sentiment_period_values(self):
        self.cb_sent_period_value.blockSignals(True)
        self.cb_sent_period_value.clear()
        self.cb_sent_period_value.addItem(PERIOD_ALL_LABEL)
        df = self.df_clean
        if df is None or df.empty or "date" not in df.columns:
            self.cb_sent_period_value.blockSignals(False)
            return
        dates = df["date"].dropna()
        if dates.empty:
            self.cb_sent_period_value.blockSignals(False)
            return
        view = self.cb_sent_view.currentText()
        if view == "월별":
            values = sorted({val.strftime("%Y-%m") for val in dates})
        elif view == "주별":
            values = sorted({val.to_period("W").start_time.strftime("%Y-%m-%d") for val in dates})
        elif view == "일별":
            values = sorted({val.strftime("%Y-%m-%d") for val in dates})
        else:
            values = sorted({str(val.year) for val in dates})
        for value in values:
            self.cb_sent_period_value.addItem(value)
        self.cb_sent_period_value.blockSignals(False)

    def create_monthly_sampling_checkbox(self):
        checkbox = QCheckBox("월별 랜덤샘플링")
        checkbox.toggled.connect(self.set_monthly_sampling)
        self.monthly_sampling_checkboxes.append(checkbox)
        return checkbox

    def set_monthly_sampling(self, checked: bool):
        self.monthly_sampling_enabled = checked
        for checkbox in self.monthly_sampling_checkboxes:
            if checkbox.isChecked() != checked:
                checkbox.blockSignals(True)
                checkbox.setChecked(checked)
                checkbox.blockSignals(False)
        self.refresh_token_sample()
        self.statusBar().showMessage(
            "월별 랜덤 샘플링 적용" if checked else "월별 랜덤 샘플링 해제"
        )

    def apply_monthly_sampling(self, df: pd.DataFrame):
        if df is None or df.empty or not self.monthly_sampling_enabled:
            return df
        if "date" not in df.columns:
            return df
        working = df.copy()
        working["_month_bucket"] = working["date"].dt.to_period("M").dt.start_time
        counts = working["_month_bucket"].value_counts()
        if counts.empty:
            return df
        target = int(counts.min())

        def sample_group(group):
            if len(group) <= target:
                return group
            return group.sample(n=target)

        sampled = working.groupby("_month_bucket", group_keys=False).apply(sample_group)
        return sampled.drop(columns=["_month_bucket"])

    def filter_df_by_period(self, df: pd.DataFrame, unit_combo: QComboBox, value_combo: QComboBox):
        if df is None or df.empty or "date" not in df.columns:
            return df
        unit = unit_combo.currentText()
        value = value_combo.currentText()
        if value in {PERIOD_ALL_LABEL, "전체"}:
            return df
        dates = df["date"]
        if unit == "연도":
            return df[dates.dt.year.astype(str) == value]
        if unit == "월":
            return df[dates.dt.strftime("%Y-%m") == value]
        if unit == "주":
            return df[dates.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d") == value]
        return df[dates.dt.strftime("%Y-%m-%d") == value]

    def filter_sentiment_by_period(self, df: pd.DataFrame):
        if df is None or df.empty or "date" not in df.columns:
            return df
        value = self.cb_sent_period_value.currentText()
        if value in {PERIOD_ALL_LABEL, "전체"}:
            return df
        view = self.cb_sent_view.currentText()
        if view == "월별":
            return df[df["date"].dt.strftime("%Y-%m") == value]
        if view == "주별":
            return df[df["date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d") == value]
        if view == "일별":
            return df[df["date"].dt.strftime("%Y-%m-%d") == value]
        return df[df["date"].dt.strftime("%Y") == value]

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "엑셀 파일 선택", "", "Excel Files (*.xlsx *.xls)"
        )
        if not file_path:
            return
        self.df_raw = pd.read_excel(file_path)
        self.lbl_file_path.setText(file_path)
        self.populate_page_type_filters()
        self.populate_language_column_options()
        self.update_language_value_list()
        self.update_mapping_status()
        self.update_preview(self.df_raw)

    def update_mapping_status(self):
        if self.df_raw is None:
            self.lbl_mapping_status.setText("컬럼 매핑 상태: -")
            return
        mapping = self.map_columns(self.df_raw)
        missing = [key for key, value in mapping.items() if value is None]
        if missing:
            self.lbl_mapping_status.setText(f"❌ 누락 컬럼: {', '.join(missing)}")
        else:
            self.lbl_mapping_status.setText("✅ 컬럼 매핑 완료")

    def populate_page_type_filters(self):
        self.list_page_type.clear()
        self.populate_sentiment_page_type_filter()
        if self.df_raw is None:
            return
        mapping = self.map_columns(self.df_raw)
        page_col = mapping.get("page_type")
        if page_col is None:
            return
        unique_vals = sorted({str(val) for val in self.df_raw[page_col].dropna().unique()})
        for val in unique_vals:
            item = QListWidgetItem(val)
            item.setCheckState(Qt.Unchecked)
            self.list_page_type.addItem(item)
        self.populate_sentiment_page_type_filter()

    def populate_sentiment_page_type_filter(self):
        if not hasattr(self, "cb_sent_page_type"):
            return
        self.cb_sent_page_type.blockSignals(True)
        self.cb_sent_page_type.clear()
        self.cb_sent_page_type.addItem("전체")
        source_df = self.df_clean if self.df_clean is not None else self.df_raw
        if source_df is not None and "page_type" in source_df.columns:
            unique_vals = sorted({str(val) for val in source_df["page_type"].dropna().unique()})
            for val in unique_vals:
                self.cb_sent_page_type.addItem(val)
        self.cb_sent_page_type.blockSignals(False)

    def populate_group_by_options(self):
        if not hasattr(self, "cb_group_by"):
            return
        source_df = self.df_clean if self.df_clean is not None else self.df_raw
        self.cb_group_by.blockSignals(True)
        self.cb_group_filter.blockSignals(True)
        self.cb_group_by.clear()
        self.cb_group_filter.clear()
        self.cb_group_filter.addItem("전체")
        if source_df is None:
            self.cb_group_by.addItem("page_type")
            self.cb_group_by.blockSignals(False)
            self.cb_group_filter.blockSignals(False)
            return
        excluded = {"date", "full_text", "full_text_en"}
        candidates = [col for col in source_df.columns if col not in excluded]
        if not candidates:
            candidates = ["page_type"]
        for col in candidates:
            self.cb_group_by.addItem(str(col))
        self.cb_group_by.blockSignals(False)
        self.cb_group_filter.blockSignals(False)
        self.update_group_filter_values()

    def update_group_filter_values(self):
        if not hasattr(self, "cb_group_filter") or self.df_clean is None:
            return
        group_col = self.cb_group_by.currentText()
        self.cb_group_filter.blockSignals(True)
        self.cb_group_filter.clear()
        self.cb_group_filter.addItem("전체")
        if group_col and group_col in self.df_clean.columns:
            unique_vals = sorted({str(val) for val in self.df_clean[group_col].dropna().unique()})
            for val in unique_vals:
                self.cb_group_filter.addItem(val)
        self.cb_group_filter.blockSignals(False)

    def populate_language_column_options(self):
        self.cb_language_column.blockSignals(True)
        current = self.cb_language_column.currentText()
        self.cb_language_column.clear()
        self.cb_language_column.addItem("None")
        if self.df_raw is not None:
            for col in self.df_raw.columns:
                self.cb_language_column.addItem(str(col))
        if current:
            index = self.cb_language_column.findText(current)
            if index >= 0:
                self.cb_language_column.setCurrentIndex(index)
        self.cb_language_column.blockSignals(False)

    def select_all_page_types(self):
        for idx in range(self.list_page_type.count()):
            self.list_page_type.item(idx).setCheckState(Qt.Checked)

    def clear_all_page_types(self):
        for idx in range(self.list_page_type.count()):
            self.list_page_type.item(idx).setCheckState(Qt.Unchecked)

    def handle_base_language_change(self):
        self.base_language = "ko" if self.rb_base_lang_ko.isChecked() else "en"
        self.sync_sentiment_language_controls()
        self.update_active_text_column()
        if self.df_clean is not None:
            self.update_preview(self.df_clean)

    def sync_sentiment_language_controls(self):
        if not hasattr(self, "rb_sent_lang_ko"):
            return
        self.rb_sent_lang_ko.blockSignals(True)
        self.rb_sent_lang_en.blockSignals(True)
        self.rb_sent_lang_ko.setChecked(self.base_language == "ko")
        self.rb_sent_lang_en.setChecked(self.base_language == "en")
        self.rb_sent_lang_ko.blockSignals(False)
        self.rb_sent_lang_en.blockSignals(False)
        self.group_sent_lang.setEnabled(False)

    def update_language_value_list(self):
        self.list_language_values.blockSignals(True)
        self.list_language_values.clear()
        self.language_column = self.get_language_column()
        if self.df_raw is None or self.language_column is None:
            self.lbl_lang_values.setText("유니크 값: -")
            self.list_language_values.blockSignals(False)
            return
        unique_vals = sorted({str(val) for val in self.df_raw[self.language_column].dropna().unique()})
        for val in unique_vals:
            item = QListWidgetItem(val)
            item.setCheckState(Qt.Unchecked)
            self.list_language_values.addItem(item)
        self.lbl_lang_values.setText(f"유니크 값: {len(unique_vals)}")
        self.list_language_values.blockSignals(False)
        self.apply_language_filter_preview()

    def select_all_languages(self):
        for idx in range(self.list_language_values.count()):
            self.list_language_values.item(idx).setCheckState(Qt.Checked)
        self.apply_language_filter_preview()

    def clear_all_languages(self):
        for idx in range(self.list_language_values.count()):
            self.list_language_values.item(idx).setCheckState(Qt.Unchecked)
        self.apply_language_filter_preview()

    def apply_language_filter_preview(self):
        self.language_filter_active = self.chk_lang_filter_active.isChecked()
        self.language_filter_values = self.get_selected_language_values()
        if self.df_raw is not None:
            self.update_preview(self.df_raw)

    def get_selected_language_values(self) -> set[str]:
        selected = set()
        for idx in range(self.list_language_values.count()):
            item = self.list_language_values.item(idx)
            if item.checkState() == Qt.Checked:
                selected.add(item.text())
        return selected

    def get_language_column(self) -> str | None:
        if not hasattr(self, "cb_language_column"):
            return None
        col = self.cb_language_column.currentText()
        return None if col == "None" else col

    def edit_language_mapping(self):
        current = json.dumps(self.country_language_map, ensure_ascii=False, indent=2)
        text, ok = QInputDialog.getMultiLineText(
            self, "Country → Language 매핑 편집", "JSON 형식으로 입력하세요:", current
        )
        if not ok:
            return
        try:
            updated = json.loads(text)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "형식 오류", "JSON 형식이 올바르지 않습니다.")
            return
        if not isinstance(updated, dict):
            QMessageBox.warning(self, "형식 오류", "JSON 객체(dict) 형태로 입력해주세요.")
            return
        self.country_language_map = {str(k).upper(): str(v).lower() for k, v in updated.items()}
        self.statusBar().showMessage("매핑 테이블을 업데이트했습니다.")

    def normalize_language_value(self, value: str) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            return None
        for sep in ["-", "_"]:
            if sep in value:
                value = value.split(sep, 1)[0]
        if value.lower() in SUPPORTED_LANG_CODES:
            return value.lower()
        upper = value.upper()
        if upper in self.country_language_map:
            return self.country_language_map[upper]
        lower = value.lower()
        if lower in SUPPORTED_LANG_CODES:
            return lower
        return None

    def apply_language_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or self.language_column is None:
            return df
        if not self.language_filter_active or not self.language_filter_values:
            return df
        return df[df[self.language_column].astype(str).isin(self.language_filter_values)]

    def update_active_text_column(self):
        if self.base_language == "en" and self.df_clean is not None:
            if "full_text_en" in self.df_clean.columns:
                self.text_column = "full_text_en"
                return
        self.text_column = "full_text"

    def get_analysis_text_column(self, df: pd.DataFrame) -> str:
        if df is not None and self.text_column in df.columns:
            return self.text_column
        return "full_text"

    def get_argos_model_paths(self) -> list[Path]:
        return [
            Path(resource_path(DEFAULT_ARGOS_MODELS_DIR)),
            DEFAULT_RESOURCE_DIR / DEFAULT_ARGOS_MODELS_DIR,
            Path(__file__).resolve().parent / DEFAULT_ARGOS_MODELS_DIR,
        ]

    def find_argos_model_path(self, source_lang: str, target_lang: str = "en") -> Path | None:
        pattern = f"{source_lang}_{target_lang}"
        for base_dir in self.get_argos_model_paths():
            if not base_dir.exists():
                continue
            for path in base_dir.glob("*.argosmodel"):
                if pattern in path.stem:
                    return path
        return None

    def translate_selected_languages(self):
        if self.df_clean is None:
            self.statusBar().showMessage("전처리 적용 후 번역을 실행해주세요.")
            return
        if importlib.util.find_spec("argostranslate") is None:
            QMessageBox.warning(self, "번역 엔진 없음", "Argos Translate가 설치되어 있지 않습니다.")
            return
        from argostranslate import package as argos_package
        from argostranslate import translate as argos_translate

        self.language_column = self.get_language_column()
        if self.language_column is None:
            QMessageBox.warning(self, "언어 컬럼 없음", "Language column을 선택해주세요.")
            return
        selected_values = self.get_selected_language_values()
        if not selected_values:
            QMessageBox.warning(self, "선택 없음", "번역할 언어 값을 선택해주세요.")
            return

        value_to_lang = {
            value: self.normalize_language_value(value) for value in selected_values
        }
        missing = [val for val, code in value_to_lang.items() if code is None]
        if missing:
            QMessageBox.warning(
                self,
                "매핑 실패",
                f"언어 코드로 매핑되지 않은 값: {', '.join(missing)}",
            )
        value_to_lang = {val: code for val, code in value_to_lang.items() if code}
        if not value_to_lang:
            return

        translators = {}
        for code in set(value_to_lang.values()):
            if code == "en":
                continue
            model_path = self.find_argos_model_path(code, "en")
            if model_path is None:
                QMessageBox.warning(
                    self,
                    "모델 누락",
                    f"{code} → en 모델을 찾을 수 없습니다.\n"
                    f"폴더: {', '.join(str(p) for p in self.get_argos_model_paths())}",
                )
                return
            argos_package.install_from_path(str(model_path))
            translator = argos_translate.get_translation_from_codes(code, "en")
            if translator is None:
                QMessageBox.warning(self, "번역기 초기화 실패", f"{code} → en 번역기 생성 실패")
                return
            translators[code] = translator

        df = self.df_clean.copy()
        target_col = "full_text" if self.chk_translate_overwrite.isChecked() else "full_text_en"
        if target_col not in df.columns:
            df[target_col] = df["full_text"]
        mask = df[self.language_column].astype(str).isin(selected_values)
        subset = df.loc[mask, ["full_text", self.language_column]]

        translated_texts = []
        for _, row in subset.iterrows():
            lang_code = value_to_lang.get(str(row[self.language_column]))
            text = row["full_text"]
            if lang_code == "en" or not isinstance(text, str):
                translated_texts.append(text)
                continue
            translator = translators.get(lang_code)
            translated_texts.append(translator.translate(text) if translator else text)

        df.loc[mask, target_col] = translated_texts
        self.df_clean = df
        self.update_active_text_column()
        self.update_preview(self.df_clean)
        self.statusBar().showMessage("선택한 언어 번역을 완료했습니다.")

    def map_columns(self, df):
        normalized = {normalize_column_name(col): col for col in df.columns}
        mapping = {
            "date": None,
            "page_type": None,
            "full_text": None,
        }
        for key in ["date", "data"]:
            if key in normalized:
                mapping["date"] = normalized[key]
                break
        for key in ["pagetype"]:
            if key in normalized:
                mapping["page_type"] = normalized[key]
                break
        for key in [
            "fulltext",
            "full_text",
            "content",
            "text",
        ]:
            if key in normalized:
                mapping["full_text"] = normalized[key]
                break
        return mapping

    def apply_cleaning(self):
        if self.df_raw is None:
            self.statusBar().showMessage("엑셀 파일을 먼저 불러주세요.")
            return
        mapping = self.map_columns(self.df_raw)
        missing = [key for key, value in mapping.items() if value is None]
        if missing:
            self.statusBar().showMessage(f"필수 컬럼 누락: {', '.join(missing)}")
            return
        df = self.apply_language_filter(self.df_raw.copy())
        df_mapped = df.copy()
        df_mapped["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce").dt.normalize()
        df_mapped["page_type"] = df[mapping["page_type"]].astype(str)
        df_mapped["full_text"] = df[mapping["full_text"]].astype(str)

        excluded_page_types = {
            self.list_page_type.item(idx).text()
            for idx in range(self.list_page_type.count())
            if self.list_page_type.item(idx).checkState() == Qt.Checked
        }
        if excluded_page_types:
            df_mapped = df_mapped[~df_mapped["page_type"].isin(excluded_page_types)]

        keywords_raw = self.le_exclude_keywords.text().strip()
        if keywords_raw:
            keywords = [kw.strip().lower() for kw in re.split(r"[|,]", keywords_raw) if kw.strip()]
            if keywords:
                mask = df_mapped["full_text"].str.lower().apply(
                    lambda text: any(kw in text for kw in keywords)
                )
                df_mapped = df_mapped[~mask]

        self.df_clean = df_mapped
        self.lbl_rows.setText(f"원본 {len(self.df_raw)} → 현재 {len(self.df_clean)}")
        self.update_active_text_column()
        self.update_preview(self.df_clean)
        self.populate_sentiment_page_type_filter()
        self.populate_group_by_options()
        self.update_gate_state()

    def update_preview(self, df):
        if df is None or df.empty:
            self.tbl_preview.setRowCount(0)
            return
        preview_df = self.apply_language_filter(df)
        preview = preview_df.head(200)
        date_col, page_col, text_col = self.get_preview_columns(preview_df)
        headers = ["date", "page_type", text_col]
        self.tbl_preview.setColumnCount(3)
        self.tbl_preview.setHorizontalHeaderLabels(headers)
        self.tbl_preview.setRowCount(len(preview))
        for row_idx, (_, row) in enumerate(preview.iterrows()):
            self.tbl_preview.setItem(
                row_idx, 0, QTableWidgetItem(self.format_date(row.get(date_col)))
            )
            self.tbl_preview.setItem(row_idx, 1, QTableWidgetItem(str(row.get(page_col, ""))))
            self.tbl_preview.setItem(row_idx, 2, QTableWidgetItem(str(row.get(text_col, ""))))

    def get_preview_columns(self, df: pd.DataFrame) -> tuple[str, str, str]:
        if df is None:
            return ("date", "page_type", self.text_column)
        if "date" in df.columns and "page_type" in df.columns:
            text_col = self.text_column if self.text_column in df.columns else "full_text"
            return ("date", "page_type", text_col)
        mapping = self.map_columns(df)
        date_col = mapping.get("date") or "date"
        page_col = mapping.get("page_type") or "page_type"
        text_col = mapping.get("full_text") or "full_text"
        return (date_col, page_col, text_col)

    def format_date(self, value):
        return safe_strftime(value)

    def apply_brand_dict(self):
        topic_name = self.le_topic_name.text().strip()
        related_raw = self.txt_topic_related.toPlainText()
        related = [kw.strip() for kw in re.split(r"[,\n]", related_raw) if kw.strip()]
        keywords = [kw for kw in related if kw]
        if topic_name and keywords:
            self.brand_map[topic_name] = keywords
            self.le_topic_name.clear()
            self.txt_topic_related.clear()
        self.refresh_topic_list()
        self.statusBar().showMessage("토픽 사전을 추가했습니다.")

    def refresh_topic_list(self):
        self.list_topics.clear()
        for topic, keywords in self.brand_map.items():
            self.list_topics.addItem(f"{topic}: {', '.join(keywords)}")
        self.cb_brand_filter.blockSignals(True)
        self.cb_brand_filter.clear()
        self.cb_brand_filter.addItem("전체")
        for brand in sorted(self.brand_map.keys()):
            self.cb_brand_filter.addItem(brand)
        self.cb_brand_filter.addItem("기타")
        self.cb_brand_filter.blockSignals(False)

    def remove_selected_topic(self):
        item = self.list_topics.currentItem()
        if item is None:
            return
        text = item.text()
        topic = text.split(":", 1)[0].strip()
        if topic in self.brand_map:
            self.brand_map.pop(topic)
        self.refresh_topic_list()

    def save_topic_dictionary(self):
        if not self.brand_map:
            self.statusBar().showMessage("저장할 토픽이 없습니다.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "토픽 사전 저장",
            "",
            "JSON Files (*.json);;Text Files (*.txt)",
        )
        if not file_path:
            return
        if file_path.endswith(".txt"):
            lines = [f"{topic}:{', '.join(words)}" for topic, words in self.brand_map.items()]
            payload = "\n".join(lines)
        else:
            payload = json.dumps(self.brand_map, ensure_ascii=False, indent=2)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(payload)
        self.statusBar().showMessage(f"토픽 사전 저장 완료: {file_path}")

    def load_topic_dictionary(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "토픽 사전 불러오기",
            "",
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        loaded = {}
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                loaded = {str(k): list(v) for k, v in data.items()}
        except json.JSONDecodeError:
            loaded = parse_brand_dictionary(content)
        if not loaded:
            self.statusBar().showMessage("토픽 사전을 불러오지 못했습니다.")
            return
        for topic, keywords in loaded.items():
            self.brand_map[topic] = [kw for kw in keywords if kw]
        self.refresh_topic_list()
        self.statusBar().showMessage(f"토픽 사전 불러오기 완료: {len(loaded)}개")

    def split_sentiment_sentences(self, text: str):
        if not isinstance(text, str):
            return []
        mode = self.cb_sentence_split.currentText()
        pattern = r"[.!?。！？\n]+"
        if mode.startswith("강함"):
            pattern = r"[.!?。！？\n,]+"
        parts = re.split(pattern, text)
        sentences = []
        for part in parts:
            sentence = part.strip()
            if not sentence or len(sentence) < 3:
                continue
            if not re.search(r"[A-Za-z0-9가-힣]", sentence):
                continue
            sentences.append(sentence)
        return sentences

    def update_sentiment_lexicon(self, include_english: bool = False):
        if self.senti_dict is None:
            self.senti_dict = load_knu_dictionary(self)
        if include_english and self.senti_dict_en is None:
            self.senti_dict_en = load_english_dictionary(self)
        self.senti_max_n = self.calculate_max_ngram(self.senti_dict)
        self.senti_max_n_en = self.calculate_max_ngram(self.senti_dict_en)

    def calculate_max_ngram(self, senti_dict):
        max_n = 1
        if not senti_dict:
            return max_n
        for word in senti_dict.keys():
            token_len = max(1, len(str(word).split()))
            if token_len > max_n:
                max_n = token_len
        return max_n

    def handle_sentiment_language_change(self, checked: bool | None = None):
        if self.df_clean is not None and self.sentiment_records_df is not None:
            self.run_sentiment()
            return
        self.update_sentiment_view()

    def ensure_topic_dictionary(self, show_warning: bool = True) -> bool:
        if self.brand_map:
            return True
        if show_warning:
            QMessageBox.warning(
                self,
                "토픽 사전 없음",
                "사전별 감성을 보려면 먼저 토픽 사전을 추가하세요.",
            )
        return False

    def match_sentiment_tokens(self, tokens, senti_dict, max_n):
        if not tokens or not senti_dict:
            return []
        used = [False] * len(tokens)
        matched = []
        for n in range(max_n, 0, -1):
            if len(tokens) < n:
                continue
            idx = 0
            while idx <= len(tokens) - n:
                if any(used[idx : idx + n]):
                    idx += 1
                    continue
                cand1 = "".join(tokens[idx : idx + n]).strip()
                cand2 = " ".join(tokens[idx : idx + n]).strip()
                hit = None
                if cand1 in senti_dict:
                    hit = cand1
                elif cand2 in senti_dict:
                    hit = cand2
                if hit:
                    matched.append({"token": hit, "start": idx, "end": idx + n})
                    for pos in range(idx, idx + n):
                        used[pos] = True
                    idx += n
                else:
                    idx += 1
        return matched

    def apply_negation_scope(self, tokens, matches, senti_dict, negation_tokens):
        adjusted_scores = []
        negated_tokens = set()
        for match in matches:
            token = match["token"]
            start = match["start"]
            score = senti_dict.get(token, 0)
            scope_start = max(0, start - 3)
            scope_tokens = tokens[scope_start:start]
            if any(scope in negation_tokens for scope in scope_tokens):
                score = -score
                negated_tokens.add(token)
            adjusted_scores.append(score)
        return adjusted_scores, negated_tokens

    def split_contrast_clauses(self, sentence: str):
        for token in CONTRAST_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", sentence, flags=re.IGNORECASE):
                parts = re.split(rf"\b{re.escape(token)}\b", sentence, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    before, after = parts
                    return [before.strip(), after.strip()]
            if re.fullmatch(r"[A-Za-z]+", token):
                continue
            if token in sentence:
                before, after = sentence.split(token, 1)
                return [before.strip(), after.strip()]
        match = re.search(r"(지만|는데)", sentence)
        if match:
            idx = match.start()
            before = sentence[: idx + len(match.group())]
            after = sentence[idx + len(match.group()):]
            return [before.strip(), after.strip()]
        return [sentence]

    def calculate_sentence_score(
        self,
        sentence: str,
        tokens,
        matches,
        apply_neg_scope: bool,
        senti_dict,
        negation_tokens,
    ):
        matched_tokens = [match["token"] for match in matches]
        reasons = []
        if not matches:
            return 0, matched_tokens, reasons
        if apply_neg_scope:
            scores, negated = self.apply_negation_scope(tokens, matches, senti_dict, negation_tokens)
            if negated:
                reasons.append("부정어 스코프")
        else:
            scores = [senti_dict.get(match["token"], 0) for match in matches]
        raw_score = sum(scores)
        return raw_score, matched_tokens, reasons

    def bin_sentiment_score(self, raw_score: float) -> int:
        if raw_score <= -1.5:
            return -2
        if raw_score < -0.5:
            return -1
        if -0.5 <= raw_score <= 0.5:
            return 0
        if raw_score < 1.5:
            return 1
        return 2

    def sentiment_score_label(self, score: int) -> str:
        return {
            -2: "가장 나쁨",
            -1: "나쁨",
            0: "중립",
            1: "좋음",
            2: "가장 좋음",
        }.get(score, "중립")

    def sentiment_bucket_label(self, score: int) -> str:
        return {
            -2: "매우부정",
            -1: "부정",
            0: "중립",
            1: "긍정",
            2: "매우긍정",
        }.get(score, str(score))

    def match_topic(self, text: str) -> str:
        if not isinstance(text, str):
            return "기타"
        if not self.brand_map:
            return "기타"
        lowered = text.lower()
        for topic, keywords in self.brand_map.items():
            for keyword in keywords:
                if keyword.lower() in lowered:
                    return topic
        return "기타"

    def apply_stopwords(self):
        raw = self.txt_stopwords.toPlainText()
        tokens = [token.strip() for token in re.split(r"[,\n]", raw) if token.strip()]
        self.stopwords = set(tokens)
        self.clean_opts = {
            "remove_numbers": self.chk_remove_numbers.isChecked(),
            "remove_symbols": self.chk_remove_symbols.isChecked(),
            "remove_single": self.chk_remove_single.isChecked(),
            "korean_only": self.chk_korean_only.isChecked(),
            "english_only": self.chk_english_only.isChecked(),
        }
        self.statusBar().showMessage("불용어/옵션을 적용했습니다.")
        self.refresh_token_sample()

    def parse_rule_table(self, table):
        rules = []
        for row in range(table.rowCount()):
            first = table.item(row, 0)
            second = table.item(row, 1) if table.columnCount() > 1 else None
            if first is None:
                continue
            left = first.text().strip()
            right = second.text().strip() if second else ""
            if left and right:
                rules.append((left, right))
        return rules

    def apply_preprocess_rules(self):
        raw_stopwords = self.txt_pre_stopwords.toPlainText()
        tokens = [token.strip() for token in re.split(r"[,\n]", raw_stopwords) if token.strip()]
        self.preprocess_stopwords = set(tokens)
        self.preprocess_length_filter = self.sb_min_length.value()
        self.statusBar().showMessage("전처리 규칙을 적용했습니다.")
        self.refresh_token_sample()
        if self.word_freq_df is not None:
            self.build_wordcloud()
        if self.graph_full is not None:
            self.build_network()
        if self.sentiment_records_df is not None:
            self.run_sentiment()

    def reset_preprocess_rules(self):
        self.cb_unicode_norm.setCurrentIndex(0)
        self.chk_trim_space.setChecked(False)
        self.chk_unify_quotes.setChecked(False)
        self.chk_model_punct.setChecked(False)
        self.chk_repeat_reduce.setChecked(False)
        self.chk_lowercase.setChecked(False)
        self.cb_contractions.setCurrentIndex(0)
        self.chk_lemmatize.setChecked(False)
        self.txt_pre_stopwords.clear()
        self.sb_min_length.setValue(0)
        self.chk_plural_singular.setChecked(False)
        for table in self.preprocess_tables.values():
            table.clearContents()
        self.statusBar().showMessage("전처리 규칙을 초기화했습니다.")
        self.refresh_token_sample()

    def save_preprocess_rules(self):
        self.statusBar().showMessage("전처리 규칙을 저장했습니다.")

    def show_preprocess_token_menu(self, pos):
        menu = QMenu(self)
        action_delete = menu.addAction("삭제")
        action_merge = menu.addAction("병합")
        action = menu.exec_(self.tbl_preprocess_tokens.viewport().mapToGlobal(pos))
        if action == action_delete:
            rows = sorted({idx.row() for idx in self.tbl_preprocess_tokens.selectionModel().selectedRows()}, reverse=True)
            for row in rows:
                self.tbl_preprocess_tokens.removeRow(row)
        elif action == action_merge:
            rows = sorted({idx.row() for idx in self.tbl_preprocess_tokens.selectionModel().selectedRows()})
            if len(rows) < 2:
                return
            tokens = []
            for row in rows:
                item = self.tbl_preprocess_tokens.item(row, 0)
                if item:
                    tokens.append(item.text())
            if not tokens:
                return
            merged, ok = QInputDialog.getText(
                self, "토큰 병합", "병합 토큰 입력", text="".join(tokens)
            )
            if not ok or not merged.strip():
                return
            self.manual_token_merges.append((tokens, merged.strip()))
            for row in reversed(rows[1:]):
                self.tbl_preprocess_tokens.removeRow(row)
            self.tbl_preprocess_tokens.setItem(rows[0], 0, QTableWidgetItem(merged.strip()))

    def save_preprocess_token_edits(self):
        replacements = {}
        exclusions = set()
        tokens_in_table = []
        for row in range(self.tbl_preprocess_tokens.rowCount()):
            token_item = self.tbl_preprocess_tokens.item(row, 0)
            count_item = self.tbl_preprocess_tokens.item(row, 1)
            if token_item is None or count_item is None:
                continue
            token = token_item.text().strip()
            original = token_item.data(Qt.UserRole)
            if original and token and token != original:
                replacements[original] = token
            if not token:
                if original:
                    exclusions.add(original)
            if token:
                tokens_in_table.append(token)
        if hasattr(self, "current_token_freq"):
            excluded = set(self.current_token_freq.index) - set(tokens_in_table)
            self.manual_token_exclusions.update(excluded)
        self.manual_token_replacements.update(replacements)
        self.manual_token_exclusions.update(exclusions)
        self.statusBar().showMessage("토큰 편집 결과를 저장했습니다.")
        self.refresh_token_sample()

    def show_wordcloud_menu(self, pos):
        menu = QMenu(self)
        action_delete = menu.addAction("삭제")
        action = menu.exec_(self.tbl_wc_topn.viewport().mapToGlobal(pos))
        if action == action_delete:
            rows = sorted({idx.row() for idx in self.tbl_wc_topn.selectionModel().selectedRows()}, reverse=True)
            for row in rows:
                self.tbl_wc_topn.removeRow(row)

    def save_wordcloud_edits(self):
        replacements = {}
        exclusions = set()
        tokens_in_table = []
        for row in range(self.tbl_wc_topn.rowCount()):
            token_item = self.tbl_wc_topn.item(row, 0)
            if token_item is None:
                continue
            token = token_item.text().strip()
            original = token_item.data(Qt.UserRole)
            if original and token and token != original:
                replacements[original] = token
            if original and not token:
                exclusions.add(original)
            if token:
                tokens_in_table.append(token)
        self.manual_token_replacements.update(replacements)
        self.manual_token_exclusions.update(exclusions)
        if self.last_wc_topn:
            excluded = set(self.last_wc_topn) - set(tokens_in_table)
            self.manual_token_exclusions.update(excluded)
        self.statusBar().showMessage("워드클라우드 테이블 편집을 저장했습니다.")
        self.build_wordcloud()

    def save_network_edits(self):
        edges = []
        for row in range(self.tbl_edges.rowCount()):
            source_item = self.tbl_edges.item(row, 0)
            target_item = self.tbl_edges.item(row, 1)
            weight_item = self.tbl_edges.item(row, 2)
            if not source_item or not target_item:
                continue
            source = source_item.text().strip()
            target = target_item.text().strip()
            if not source or not target:
                continue
            try:
                weight = float(weight_item.text()) if weight_item else 1.0
            except ValueError:
                weight = 1.0
            edges.append((source, target, weight))

        nodes = set()
        for row in range(self.tbl_nodes.rowCount()):
            node_item = self.tbl_nodes.item(row, 0)
            if not node_item:
                continue
            node = node_item.text().strip()
            if node:
                nodes.add(node)
        for source, target, _ in edges:
            nodes.add(source)
            nodes.add(target)

        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node)
        for source, target, weight in edges:
            graph.add_edge(source, target, weight=weight)

        self.graph_full = graph
        self.graph_view = graph.copy()
        self.network_pos = None
        self.network_seed_nodes = []
        self.network_level_map = {}
        self.nodes_df = pd.DataFrame(
            [(node, graph.degree(node)) for node in graph.nodes], columns=["node", "degree"]
        )
        self.edges_df = pd.DataFrame(
            [(u, v, data.get("weight", 1.0)) for u, v, data in graph.edges(data=True)],
            columns=["source", "target", "weight"],
        )
        self.nodes_view_df = self.nodes_df.copy()
        self.edges_view_df = self.edges_df.copy()
        self.populate_node_lists(list(graph.nodes))
        self.draw_network(graph)
        self.populate_network_tables(self.edges_view_df, self.nodes_view_df)
        self.statusBar().showMessage("네트워크 테이블 편집을 저장했습니다.")

    def show_sentiment_menu(self, pos):
        menu = QMenu(self)
        action_delete = menu.addAction("삭제")
        action = menu.exec_(self.tbl_sent_records.viewport().mapToGlobal(pos))
        if action == action_delete:
            rows = sorted({idx.row() for idx in self.tbl_sent_records.selectionModel().selectedRows()}, reverse=True)
            for row in rows:
                self.tbl_sent_records.removeRow(row)

    def save_sentiment_edits(self):
        if self.sentiment_records_df is None or self.sentiment_records_df.empty:
            return
        updated = self.sentiment_records_df.copy()
        drop_indices = []
        for row in range(self.tbl_sent_records.rowCount()):
            date_item = self.tbl_sent_records.item(row, 0)
            if date_item is None:
                continue
            idx = date_item.data(Qt.UserRole)
            if idx is None:
                continue
            sentence_item = self.tbl_sent_records.item(row, 2)
            if sentence_item is None or not sentence_item.text().strip():
                drop_indices.append(idx)
                continue
            page_type = self.tbl_sent_records.item(row, 1).text() if self.tbl_sent_records.item(row, 1) else ""
            sentence = sentence_item.text()
            score_text = self.tbl_sent_records.item(row, 3).text() if self.tbl_sent_records.item(row, 3) else "0"
            label = self.tbl_sent_records.item(row, 4).text() if self.tbl_sent_records.item(row, 4) else ""
            topic = self.tbl_sent_records.item(row, 5).text() if self.tbl_sent_records.item(row, 5) else ""
            try:
                score = int(float(score_text))
            except ValueError:
                score = 0
            updated.at[idx, "page_type"] = page_type
            updated.at[idx, "sentence"] = sentence
            updated.at[idx, "score"] = score
            updated.at[idx, "label"] = label
            updated.at[idx, "topic"] = topic
        if drop_indices:
            updated = updated.drop(index=drop_indices, errors="ignore")
        self.sentiment_records_df = updated
        self.update_sentiment_view()
        self.statusBar().showMessage("감성 테이블 편집을 저장했습니다.")

    def show_token_menu(self, pos):
        selection = self.tbl_token_sample.selectionModel().selectedRows()
        if selection:
            rows = [index.row() for index in selection]
        else:
            item = self.tbl_token_sample.itemAt(pos)
            if item is None:
                return
            rows = [item.row()]
        tokens = []
        for row in rows:
            token_item = self.tbl_token_sample.item(row, 0)
            if token_item is not None:
                tokens.append(token_item.text())
        if not tokens:
            return
        menu = QMenu(self)
        action_delete = menu.addAction("선택 단어 삭제(불용어 추가)")
        action = menu.exec_(self.tbl_token_sample.viewport().mapToGlobal(pos))
        if action == action_delete:
            self.add_stopwords_from_table(tokens)

    def add_stopwords_from_table(self, tokens: list[str]):
        cleaned = [token.strip() for token in tokens if token and token.strip()]
        if not cleaned:
            return
        self.stopwords.update(cleaned)
        existing = set(re.split(r"[,\n]", self.txt_stopwords.toPlainText()))
        additions = [token for token in cleaned if token not in existing]
        if additions:
            current = self.txt_stopwords.toPlainText().strip()
            updated = f"{current}\n" if current else ""
            updated += "\n".join(additions)
            self.txt_stopwords.setPlainText(updated)
        self.refresh_token_sample()
        if self.word_freq_df is not None:
            self.build_wordcloud()
        if self.graph_full is not None:
            self.build_network()

    def refresh_token_sample(self):
        if self.df_clean is None:
            return
        df = self.apply_monthly_sampling(self.df_clean)
        text_col = self.get_analysis_text_column(df)
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in df[text_col]
        ))
        series = pd.Series(tokens)
        freq = series.value_counts()
        self.current_token_freq = freq
        self.tbl_token_sample.setRowCount(len(freq))
        for row_idx, (token, count) in enumerate(freq.items()):
            token_item = QTableWidgetItem(token)
            token_item.setData(Qt.UserRole, token)
            self.tbl_token_sample.setItem(row_idx, 0, token_item)
            self.tbl_token_sample.setItem(row_idx, 1, QTableWidgetItem(str(count)))
        if hasattr(self, "tbl_preprocess_tokens"):
            self.tbl_preprocess_tokens.setRowCount(len(freq))
            for row_idx, (token, count) in enumerate(freq.items()):
                token_item = QTableWidgetItem(token)
                token_item.setData(Qt.UserRole, token)
                self.tbl_preprocess_tokens.setItem(row_idx, 0, token_item)
                self.tbl_preprocess_tokens.setItem(row_idx, 1, QTableWidgetItem(str(count)))
            self.lbl_preprocess_count.setText(f"토큰 {len(tokens)} / 고유 {len(freq)}")

    def apply_replacements(self, text: str, rules):
        for src, dest in rules:
            if not src:
                continue
            text = re.sub(re.escape(src), dest, text)
        return text

    def apply_preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        norm_mode = self.cb_unicode_norm.currentText() if hasattr(self, "cb_unicode_norm") else "사용 안 함"
        if norm_mode in {"NFKC", "NFC"}:
            text = unicodedata.normalize(norm_mode, text)
        if getattr(self, "chk_trim_space", None) and self.chk_trim_space.isChecked():
            text = SPACE_RE.sub(" ", text).strip()
        if getattr(self, "chk_unify_quotes", None) and self.chk_unify_quotes.isChecked():
            text = text.translate(DASH_TRANSLATION).translate(QUOTE_TRANSLATION)
        if getattr(self, "chk_model_punct", None) and self.chk_model_punct.isChecked():
            text = re.sub(r"([A-Za-z])[-_\s]+(\d)", r"\1\2", text)
            text = re.sub(r"(\d)[-_\s]+([A-Za-z])", r"\1\2", text)
        if hasattr(self, "tbl_join_map"):
            text = self.apply_replacements(text, self.parse_rule_table(self.tbl_join_map))
        if hasattr(self, "tbl_ko_en_map"):
            text = self.apply_replacements(text, self.parse_rule_table(self.tbl_ko_en_map))
        if hasattr(self, "tbl_compound_map"):
            text = self.apply_replacements(text, self.parse_rule_table(self.tbl_compound_map))
        if getattr(self, "chk_repeat_reduce", None) and self.chk_repeat_reduce.isChecked():
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        if getattr(self, "chk_lowercase", None) and self.chk_lowercase.isChecked():
            text = text.lower()
        if getattr(self, "cb_contractions", None) and self.cb_contractions.currentText().endswith("적용"):
            for src, dest in CONTRACTIONS_MAP.items():
                text = re.sub(rf"\b{re.escape(src)}\b", dest, text, flags=re.IGNORECASE)
        return text

    def get_pos_filter_tags(self):
        if getattr(self, "rb_pos_all", None) and self.rb_pos_all.isChecked():
            return {"NNG", "NNP", "NNB", "NR", "NP", "VA", "VV", "VX"}
        if getattr(self, "rb_pos_noun_adj", None) and self.rb_pos_noun_adj.isChecked():
            return {"NNG", "NNP", "NNB", "NR", "NP", "VA"}
        if getattr(self, "rb_pos_noun_verb", None) and self.rb_pos_noun_verb.isChecked():
            return {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VX"}
        return {"NNG", "NNP", "NNB", "NR", "NP"}

    def merge_token_sequences(self, tokens, merge_rules):
        if not merge_rules:
            return tokens
        idx = 0
        merged = []
        while idx < len(tokens):
            matched = False
            for seq, replacement in merge_rules:
                if tokens[idx : idx + len(seq)] == seq:
                    merged.append(replacement)
                    idx += len(seq)
                    matched = True
                    break
            if not matched:
                merged.append(tokens[idx])
                idx += 1
        return merged

    def apply_post_token_rules(self, tokens):
        if not tokens:
            return []
        map_rules = dict(self.parse_rule_table(self.tbl_repr_map)) if hasattr(self, "tbl_repr_map") else {}
        syn_rules = dict(self.parse_rule_table(self.tbl_syn_map)) if hasattr(self, "tbl_syn_map") else {}
        model_rules = dict(self.parse_rule_table(self.tbl_model_map)) if hasattr(self, "tbl_model_map") else {}
        merge_rules_raw = self.parse_rule_table(self.tbl_merge_tokens) if hasattr(self, "tbl_merge_tokens") else []
        merge_rules = []
        for src, dest in merge_rules_raw:
            seq = [part.strip() for part in re.split(r"[+ ]", src) if part.strip()]
            if seq and dest:
                merge_rules.append((seq, dest))
        merge_rules.extend(self.manual_token_merges)

        updated = []
        for token in tokens:
            if token in self.manual_token_exclusions:
                continue
            token = self.manual_token_replacements.get(token, token)
            token = map_rules.get(token, token)
            token = syn_rules.get(token, token)
            token = model_rules.get(token, token)
            if getattr(self, "chk_lemmatize", None) and self.chk_lemmatize.isChecked():
                if re.fullmatch(r"[A-Za-z]+", token):
                    if token.endswith("ing") and len(token) > 4:
                        token = token[:-3]
                    elif token.endswith("ed") and len(token) > 3:
                        token = token[:-2]
            if getattr(self, "chk_plural_singular", None) and self.chk_plural_singular.isChecked():
                if re.fullmatch(r"[A-Za-z]+", token):
                    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
                        token = token[:-1]
            updated.append(token)

        updated = self.merge_token_sequences(updated, merge_rules)

        filtered = []
        for token in updated:
            if token in self.stopwords or token in self.preprocess_stopwords:
                continue
            if self.preprocess_length_filter and len(token) <= self.preprocess_length_filter:
                continue
            if self.clean_opts.get("remove_numbers") and re.search(r"\d", token):
                continue
            if self.clean_opts.get("remove_symbols") and re.search(r"[^\w\u3131-\uD79D]", token):
                continue
            if self.clean_opts.get("remove_single") and len(token) == 1:
                continue
            if self.clean_opts.get("korean_only") and not re.fullmatch(r"[\u3131-\uD79D]+", token):
                continue
            if self.clean_opts.get("english_only") and not re.fullmatch(r"[A-Za-z]+", token):
                continue
            filtered.append(token)
        return filtered

    def tokenize_text(self, text: str):
        if not isinstance(text, str):
            return []
        if self.base_language == "en":
            return self.tokenize_sentiment_english(text)
        text = self.apply_preprocess_text(text)
        analysis = self.kiwi.analyze(text)
        if not analysis or not analysis[0][0]:
            return []
        allowed_tags = self.get_pos_filter_tags()
        tokens = [token for token, tag, _, _ in analysis[0][0] if tag in allowed_tags]
        return self.apply_post_token_rules(tokens)

    def get_sentiment_pos_tags(self):
        return {
            "NNG",
            "NNP",
            "NNB",
            "NR",
            "NP",
            "VA",
            "VV",
            "VX",
            "MAG",
            "MM",
            "IC",
        }

    def tokenize_sentiment_korean(self, text: str):
        if not isinstance(text, str):
            return []
        text = self.apply_preprocess_text(text)
        analysis = self.kiwi.analyze(text)
        if not analysis or not analysis[0][0]:
            return []
        allowed_tags = self.get_sentiment_pos_tags()
        tokens = [token for token, tag, _, _ in analysis[0][0] if tag in allowed_tags]
        return self.apply_post_token_rules(tokens)

    def tokenize_sentiment_english(self, text: str):
        if not isinstance(text, str):
            return []
        text = self.apply_preprocess_text(text)
        tokens = [match.group(0).lower() for match in EN_WORD_PATTERN.finditer(text)]
        return self.apply_post_token_rules(tokens)

    def detect_sentence_language(self, sentence: str):
        if not isinstance(sentence, str):
            return "unknown"
        ko_count = len(re.findall(r"[가-힣]", sentence))
        en_count = len(re.findall(r"[A-Za-z]", sentence))
        if ko_count and en_count:
            total = ko_count + en_count
            if ko_count / total >= 0.7:
                return "ko"
            if en_count / total >= 0.7:
                return "en"
            return "mixed"
        if ko_count:
            return "ko"
        if en_count:
            return "en"
        return "unknown"

    def get_sentiment_language_mode(self):
        return "ko" if self.base_language == "ko" else "en"

    def parse_custom_terms(self, raw_text: str):
        if not raw_text:
            return set()
        tokens = [token.strip() for token in re.split(r"[,\n]", raw_text) if token.strip()]
        return set(tokens)

    def warn_no_tokens(self, feature_name: str):
        QMessageBox.warning(
            self,
            "토큰화 오류",
            f"{feature_name}에 사용할 토큰이 없습니다.\n"
            "전처리/토큰화 설정을 확인한 뒤 다시 실행해주세요.",
        )
        self.statusBar().showMessage(f"{feature_name}에 사용할 토큰이 없습니다.")

    def build_wordcloud_mask(self, shape: str, size: int = 400):
        y, x = np.ogrid[:size, :size]
        center = (size - 1) / 2
        mask = np.zeros((size, size), dtype=np.uint8)
        if shape == "circle":
            radius = size * 0.45
            shape_mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
            mask[shape_mask] = 255
        elif shape == "diamond":
            radius = size * 0.45
            shape_mask = np.abs(x - center) + np.abs(y - center) <= radius
            mask[shape_mask] = 255
        elif shape == "triangle":
            shape_mask = y >= (size - x) * 0.8
            shape_mask &= y >= x * 0.8
            mask[shape_mask] = 255
        else:
            mask[:, :] = 255
        return mask

    def random_wordcloud_style(self):
        colormaps = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "cool",
            "spring",
            "summer",
            "autumn",
            "winter",
            "Set2",
            "tab10",
        ]
        colormap = random.choice(colormaps)
        mask = self.build_wordcloud_mask("square")
        return colormap, mask

    def build_buzz(self):
        if self.df_clean is None:
            return
        df = self.filter_df_by_period(
            self.df_clean.copy(), self.cb_buzz_period_unit, self.cb_buzz_period_value
        )
        if df.empty:
            self.buzz_canvas.ax.clear()
            self.buzz_canvas.draw()
            self.tbl_buzz.setRowCount(0)
            self.statusBar().showMessage("선택한 기간에 데이터가 없습니다.")
            return
        gran = self.cb_granularity.currentText()
        if gran == "월":
            df["bucket"] = df["date"].dt.to_period("M").dt.start_time
        elif gran == "주":
            df["bucket"] = df["date"].dt.to_period("W").dt.start_time
        elif gran == "일":
            df["bucket"] = df["date"].dt.normalize()
        else:
            df["bucket"] = df["date"].dt.to_period("Y").dt.start_time

        group_col = self.cb_group_by.currentText()
        if group_col not in df.columns:
            group_col = "group"
        selected_group = self.cb_group_filter.currentText()
        if selected_group != "전체" and group_col in df.columns:
            df = df[df[group_col].astype(str) == selected_group]

        if self.chk_split_by_group.isChecked() and group_col in df.columns:
            summary = df.groupby(["bucket", group_col]).size().reset_index(name="count")
        else:
            summary = df.groupby("bucket").size().reset_index(name="count")
            summary[group_col] = "전체"

        summary = summary.sort_values("bucket")
        self.buzz_df = summary

        metric = self.cb_buzz_metric.currentText()
        labels = [self.format_date(val) for val in summary["bucket"]]
        note_text = ""
        pivot_counts = summary.pivot_table(
            index="bucket", columns=group_col, values="count", fill_value=0
        ).sort_index()
        if self.chk_split_by_group.isChecked() and group_col in summary.columns:
            if metric == "%":
                pivot = pivot_counts.div(pivot_counts.sum(axis=1).replace(0, 1), axis=0) * 100
                ylabel = "%"
            else:
                pivot = pivot_counts
                ylabel = "count"
            labels = [self.format_date(val) for val in pivot_counts.index]
            series = [(str(col), pivot[col].tolist()) for col in pivot_counts.columns]
            count_series = [pivot_counts[col].tolist() for col in pivot_counts.columns]
            self.buzz_canvas.plot_stacked_bar(
                labels,
                series,
                "버즈량",
                ylabel,
                count_series=count_series,
                note_text=note_text,
                show_labels=False,
            )
        else:
            count_values = summary["count"].tolist()
            if metric == "%":
                total = sum(count_values) or 1
                values = [round(val / total * 100, 2) for val in count_values]
                ylabel = "%"
            else:
                values = count_values
                ylabel = "count"
            self.buzz_canvas.plot_bar(
                labels,
                values,
                "버즈량",
                ylabel,
                count_values=count_values,
                note_text=note_text,
                show_labels=False,
            )
        self.chart_images["buzz"] = self.save_chart(self.buzz_canvas, "buzz")
        self.buzz_bucket_labels = labels
        self.buzz_bucket_dates = list(summary["bucket"])

        if metric == "%":
            display_df = pivot_counts.div(pivot_counts.sum(axis=1).replace(0, 1), axis=0) * 100
            display_df = display_df.round(2)
        else:
            display_df = pivot_counts
        columns = [str(col) for col in display_df.columns]
        self.tbl_buzz.setColumnCount(len(columns) + 1)
        self.tbl_buzz.setHorizontalHeaderLabels(["기간"] + columns)
        self.tbl_buzz.setRowCount(len(display_df))
        for row_idx, (bucket, row) in enumerate(display_df.iterrows()):
            self.tbl_buzz.setItem(row_idx, 0, QTableWidgetItem(self.format_date(bucket)))
            for col_idx, col in enumerate(columns, start=1):
                value = row.get(col, 0)
                self.tbl_buzz.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def on_buzz_click(self, event):
        if self.df_clean is None or event.xdata is None:
            return
        if not hasattr(self, "buzz_bucket_labels"):
            return
        idx = int(round(event.xdata))
        if idx < 0 or idx >= len(self.buzz_bucket_labels):
            return
        bucket = self.buzz_bucket_dates[idx]
        df = self.df_clean.copy()
        gran = self.cb_granularity.currentText()
        if gran == "월":
            df["bucket"] = df["date"].dt.to_period("M").dt.start_time
        elif gran == "주":
            df["bucket"] = df["date"].dt.to_period("W").dt.start_time
        elif gran == "일":
            df["bucket"] = df["date"].dt.normalize()
        else:
            df["bucket"] = df["date"].dt.to_period("Y").dt.start_time
        df = df[df["bucket"] == bucket]
        group_col = self.cb_group_by.currentText()
        if group_col not in df.columns:
            group_col = "group"
        selected_group = self.cb_group_filter.currentText()
        if selected_group != "전체" and group_col in df.columns:
            df = df[df[group_col].astype(str) == selected_group]

        if df.empty:
            return

        text_col = self.get_analysis_text_column(df)
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in df[text_col]
        ))
        freq = pd.Series(tokens).value_counts()
        hot_topics = freq.head(10).index.tolist()

        prev_bucket = None
        if idx > 0:
            prev_bucket = self.buzz_bucket_dates[idx - 1]
        hot_delta = []
        if prev_bucket is not None:
            prev_df = self.df_clean.copy()
            if gran == "월":
                prev_df["bucket"] = prev_df["date"].dt.to_period("M").dt.start_time
            elif gran == "주":
                prev_df["bucket"] = prev_df["date"].dt.to_period("W").dt.start_time
            elif gran == "일":
                prev_df["bucket"] = prev_df["date"].dt.normalize()
            else:
                prev_df["bucket"] = prev_df["date"].dt.to_period("Y").dt.start_time
            prev_df = prev_df[prev_df["bucket"] == prev_bucket]
            if selected_group != "전체" and group_col in prev_df.columns:
                prev_df = prev_df[prev_df[group_col].astype(str) == selected_group]
            prev_tokens = list(itertools.chain.from_iterable(
                self.tokenize_text(text) for text in prev_df[text_col]
            ))
            prev_freq = pd.Series(prev_tokens).value_counts()
            delta = (freq - prev_freq).fillna(0).sort_values(ascending=False)
            hot_delta = delta.head(10).index.tolist()

        voc_samples = df[text_col].head(3).tolist()
        self.lbl_buzz_filters.setText(
            f"기간/그룹: {self.format_date(bucket)} / {selected_group}"
        )
        self.txt_buzz_hot.setPlainText("\n".join(hot_delta or hot_topics))
        self.txt_buzz_top.setPlainText("\n".join(hot_topics))
        self.txt_buzz_voc.setPlainText("\n".join(voc_samples))

    def build_wordcloud(self):
        if self.df_clean is None:
            return
        try:
            topn = int(self.cb_wc_topn.currentText())
        except (TypeError, ValueError):
            topn = 0
        if topn <= 0:
            self.lbl_wc_count.setText("토큰 0 / 고유 0")
            self.lbl_wc_view.setText("워드클라우드 단어 수는 1 이상이어야 합니다")
            self.tbl_wc_topn.setRowCount(0)
            self.statusBar().showMessage("워드클라우드 단어 수를 1 이상으로 설정하세요.")
            return
        df = self.filter_df_by_period(
            self.df_clean, self.cb_wc_period_unit, self.cb_wc_period_value
        )
        df = self.apply_monthly_sampling(df)
        if df is None or df.empty:
            self.lbl_wc_count.setText("토큰 0 / 고유 0")
            self.lbl_wc_view.setText("표시할 데이터가 없습니다")
            self.tbl_wc_topn.setRowCount(0)
            self.statusBar().showMessage("선택한 기간에 워드클라우드 데이터가 없습니다.")
            return
        text_col = self.get_analysis_text_column(df)
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in df[text_col]
        ))
        if not tokens:
            self.lbl_wc_count.setText("토큰 0 / 고유 0")
            self.lbl_wc_view.setText("토큰화된 데이터가 없습니다")
            self.tbl_wc_topn.setRowCount(0)
            self.warn_no_tokens("워드클라우드")
            return
        series = pd.Series(tokens)
        freq = series.value_counts()
        self.word_freq_df = freq.reset_index().rename(columns={"index": "token", 0: "count"})
        self.lbl_wc_count.setText(f"토큰 {len(tokens)} / 고유 {len(freq)}")

        if freq.empty:
            self.lbl_wc_view.setText("데이터가 없습니다")
            self.tbl_wc_topn.setRowCount(0)
            return
        font_path = self.font_path
        if self.chk_wc_random_style.isChecked():
            colormap, mask = self.random_wordcloud_style()
        else:
            colormap, mask = "Blues", self.build_wordcloud_mask("square")
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            font_path=str(font_path) if font_path else None,
            colormap=colormap,
            mask=mask,
            repeat=True,
            prefer_horizontal=1.0,
            collocations=False,
        )
        top_freq = freq.head(topn)
        if top_freq.empty:
            self.lbl_wc_view.setText("워드클라우드를 만들 단어가 없습니다")
            self.tbl_wc_topn.setRowCount(0)
            self.statusBar().showMessage("워드클라우드에 사용할 단어가 없습니다.")
            return
        try:
            wc_img = wordcloud.generate_from_frequencies(top_freq.to_dict())
        except Exception:
            fallback_wordcloud = WordCloud(
                width=800,
                height=500,
                background_color="white",
                font_path=str(font_path) if font_path else None,
                colormap=colormap,
                repeat=True,
                prefer_horizontal=1.0,
                collocations=False,
            )
            try:
                wc_img = fallback_wordcloud.generate_from_frequencies(top_freq.to_dict())
                self.statusBar().showMessage("마스크 없이 워드클라우드를 생성했습니다.")
            except Exception as fallback_exc:
                self.lbl_wc_view.setText("워드클라우드를 생성할 수 없습니다")
                self.tbl_wc_topn.setRowCount(0)
                QMessageBox.warning(
                    self,
                    "워드클라우드 오류",
                    "워드클라우드를 생성할 수 없습니다.\n"
                    "단어 수를 줄이거나 마스크를 변경한 뒤 다시 시도해주세요.\n"
                    f"원인: {fallback_exc}",
                )
                self.statusBar().showMessage("워드클라우드 생성에 실패했습니다.")
                return
        wc_path = os.path.join(os.getcwd(), "wordcloud.png")
        wc_img.to_file(wc_path)
        self.wc_image_path = wc_path
        self.chart_images["wordcloud"] = wc_path

        pixmap = QPixmap(wc_path)
        self.lbl_wc_view.setPixmap(pixmap.scaled(self.lbl_wc_view.size(), Qt.KeepAspectRatio))

        self.last_wc_topn = list(top_freq.index)
        self.last_wc_topn_value = topn
        self.tbl_wc_topn.setRowCount(len(top_freq))
        for row_idx, (token, count) in enumerate(top_freq.items()):
            token_item = QTableWidgetItem(token)
            token_item.setData(Qt.UserRole, token)
            self.tbl_wc_topn.setItem(row_idx, 0, token_item)
            self.tbl_wc_topn.setItem(row_idx, 1, QTableWidgetItem(str(count)))

    def toggle_network_advanced(self):
        advanced = self.cb_mode.currentText() == "고급"
        self.cb_cooc_scope.setVisible(advanced)
        self.cb_weight_mode.setVisible(advanced)
        self.lbl_advanced_hint.setVisible(advanced)

    def handle_pmi_guard(self):
        if self.cb_weight_mode.currentText() != "PMI":
            return
        if self.sb_min_node_count.value() < 10:
            self.sb_min_node_count.setValue(10)
        if self.sb_min_edge_weight.value() < 5:
            self.sb_min_edge_weight.setValue(5)
        self.statusBar().showMessage("PMI 안정화 필터를 적용했습니다")

    def build_network(self):
        if self.df_clean is None:
            return
        df = self.filter_df_by_period(
            self.df_clean, self.cb_network_period_unit, self.cb_network_period_value
        )
        df = self.apply_monthly_sampling(df)
        if df.empty:
            self.statusBar().showMessage("선택한 기간에 네트워크 데이터가 없습니다.")
            self.graph_full = None
            self.graph_view = None
            self.network_pos = None
            self.network_seed_nodes = []
            self.network_level_map = {}
            self.draw_network(nx.Graph())
            self.populate_network_tables(pd.DataFrame(), pd.DataFrame())
            return
        if self.cb_mode.currentText() == "기본":
            scope = "문서(로우)"
            weight_mode = "count"
        else:
            scope = self.cb_cooc_scope.currentText()
            weight_mode = self.cb_weight_mode.currentText()

        self.network_stopwords = self.parse_custom_terms(self.le_network_stopwords.text())
        token_lists = []
        text_col = self.get_analysis_text_column(df)
        if scope == "문서(로우)":
            for text in df[text_col]:
                tokens = [normalize_term(token) for token in self.tokenize_text(text)]
                tokens = [token for token in tokens if token and token not in self.network_stopwords]
                if tokens:
                    token_lists.append(tokens)
        else:
            for text in df[text_col]:
                for sentence in split_sentences(text):
                    tokens = [normalize_term(token) for token in self.tokenize_text(sentence)]
                    tokens = [token for token in tokens if token and token not in self.network_stopwords]
                    if tokens:
                        token_lists.append(tokens)

        if not token_lists:
            self.warn_no_tokens("네트워크")
            self.graph_full = None
            self.graph_view = None
            self.network_pos = None
            self.network_seed_nodes = []
            self.network_level_map = {}
            self.draw_network(nx.Graph())
            self.populate_network_tables(pd.DataFrame(), pd.DataFrame())
            return

        node_counts = {}
        for tokens in token_lists:
            unique_tokens = list(dict.fromkeys(tokens))
            for token in unique_tokens:
                node_counts[token] = node_counts.get(token, 0) + 1

        min_node = self.sb_min_node_count.value()
        min_edge = self.sb_min_edge_weight.value()
        max_nodes = self.sb_max_nodes.value()
        filtered_nodes = {node for node, count in node_counts.items() if count >= min_node}
        ranked_nodes = sorted(filtered_nodes, key=lambda n: node_counts[n], reverse=True)[:max_nodes]
        ranked_nodes_set = set(ranked_nodes)
        if not ranked_nodes_set:
            self.statusBar().showMessage("필터 조건에 맞는 노드가 없습니다.")
            self.graph_full = None
            self.graph_view = None
            self.network_pos = None
            self.draw_network(nx.Graph())
            self.populate_network_tables(pd.DataFrame(), pd.DataFrame())
            return

        edge_counts = {}
        for tokens in token_lists:
            unique_tokens = [token for token in dict.fromkeys(tokens) if token in ranked_nodes_set]
            if len(unique_tokens) < 2:
                continue
            for a, b in itertools.combinations(unique_tokens, 2):
                edge = tuple(sorted((a, b)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        if weight_mode == "PMI":
            total_docs = len(token_lists)
            edge_weights = {}
            for (a, b), count in edge_counts.items():
                pa = node_counts[a] / total_docs
                pb = node_counts[b] / total_docs
                pab = count / total_docs
                edge_weights[(a, b)] = max(0.0, (pab / (pa * pb)))
        else:
            edge_weights = {edge: float(count) for edge, count in edge_counts.items()}

        filtered_edges = {
            edge: weight for edge, weight in edge_weights.items() if weight >= min_edge
        }
        self.update_network_recommendation(len(token_lists), len(node_counts))

        graph = nx.Graph()
        for node in ranked_nodes:
            graph.add_node(node, count=node_counts[node])
        for (a, b), weight in filtered_edges.items():
            graph.add_edge(a, b, weight=weight)

        self.graph_full = graph
        self.graph_view = graph.copy()
        self.network_pos = None
        self.network_seed_nodes = []
        self.network_level_map = {}
        self.nodes_df = pd.DataFrame(
            [(node, graph.degree(node)) for node in graph.nodes], columns=["node", "degree"]
        )
        self.edges_df = pd.DataFrame(
            [(u, v, data["weight"]) for u, v, data in graph.edges(data=True)],
            columns=["source", "target", "weight"],
        )
        self.nodes_view_df = self.nodes_df.copy()
        self.edges_view_df = self.edges_df.copy()

        self.populate_node_lists(ranked_nodes)
        self.draw_network(graph)
        self.populate_network_tables(self.edges_view_df, self.nodes_view_df)
        self.chart_images["network"] = self.save_chart(self.network_canvas, "network")

    def populate_node_lists(self, nodes):
        self.list_nodes_ranked.clear()
        self.last_nodes_ranked = list(nodes)
        for node in nodes:
            self.list_nodes_ranked.addItem(node)

    def toggle_node_selection(self, item):
        if item is None:
            return
        if item.isSelected():
            item.setSelected(False)
        else:
            item.setSelected(True)

    def select_node_from_search(self):
        text = self.le_node_search.text().strip()
        if not text:
            return
        matches = self.list_nodes_ranked.findItems(text, Qt.MatchExactly)
        if matches:
            self.list_nodes_ranked.clearSelection()
            matches[0].setSelected(True)
            self.list_nodes_ranked.scrollToItem(matches[0])

    def apply_hop(self):
        if self.graph_full is None:
            return
        selected = self.list_nodes_ranked.selectedItems()
        seeds = [item.text() for item in selected]
        if not seeds:
            self.statusBar().showMessage("Seed 노드를 선택해주세요.")
            return
        hop = int(self.cb_hop_depth.currentText())
        nodes = set()
        for seed in seeds:
            if seed not in self.graph_full:
                continue
            nodes.update(nx.single_source_shortest_path_length(self.graph_full, seed, cutoff=hop).keys())
        subgraph = self.graph_full.subgraph(nodes).copy()
        self.graph_view = subgraph
        self.network_pos = None
        self.network_seed_nodes = seeds
        self.network_level_map = self.compute_network_levels(subgraph, seeds)
        self.nodes_view_df = pd.DataFrame(
            [(node, subgraph.degree(node)) for node in subgraph.nodes], columns=["node", "degree"]
        )
        self.edges_view_df = pd.DataFrame(
            [(u, v, data["weight"]) for u, v, data in subgraph.edges(data=True)],
            columns=["source", "target", "weight"],
        )
        self.draw_network(subgraph)
        self.populate_network_tables(self.edges_view_df, self.nodes_view_df)
        self.chart_images["network"] = self.save_chart(self.network_canvas, "network")

    def compute_network_levels(self, graph, seeds):
        if not seeds:
            return {}
        levels = {}
        for node in graph.nodes:
            distances = []
            for seed in seeds:
                if seed in graph and nx.has_path(graph, seed, node):
                    distances.append(nx.shortest_path_length(graph, seed, node))
            if not distances:
                continue
            distance = min(distances)
            if distance <= 0:
                levels[node] = 1
            elif distance == 1:
                levels[node] = 2
            else:
                levels[node] = 3
        return levels

    def reset_network_view(self):
        if self.graph_full is None:
            return
        self.graph_view = self.graph_full.copy()
        self.network_pos = None
        self.network_seed_nodes = []
        self.network_level_map = {}
        self.nodes_view_df = self.nodes_df.copy()
        self.edges_view_df = self.edges_df.copy()
        self.draw_network(self.graph_view)
        self.populate_network_tables(self.edges_view_df, self.nodes_view_df)
        self.chart_images["network"] = self.save_chart(self.network_canvas, "network")

    def redraw_network(self):
        if self.graph_view is None:
            return
        self.draw_network(self.graph_view)

    def draw_network(self, graph):
        self.network_canvas.ax.clear()
        if graph is None or graph.number_of_nodes() == 0:
            self.network_canvas.draw()
            return
        if self.network_pos is None or set(self.network_pos.keys()) != set(graph.nodes):
            k = max(0.3, 2 / max(1, graph.number_of_nodes()) ** 0.5)
            self.network_pos = nx.spring_layout(graph, k=k, seed=42)
        pos = self.network_pos
        degrees = [graph.degree(node) for node in graph.nodes]
        max_degree = max(degrees) if degrees else 1
        sizes = [200 + 1200 * (deg / max_degree) for deg in degrees]
        node_colors = ["#f4d03f" for _ in graph.nodes]
        edge_weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.6 + 3.0 * (weight / max_weight) for weight in edge_weights]
        edge_norm = colors.Normalize(vmin=0, vmax=max_weight or 1.0)
        edge_colors = [
            colors.to_hex(cm.Greys(0.2 + 0.7 * edge_norm(weight)))
            for weight in edge_weights
        ]
        nx.draw_networkx_nodes(
            graph, pos, ax=self.network_canvas.ax, node_size=sizes, node_color=node_colors
        )
        nx.draw_networkx_edges(
            graph, pos, ax=self.network_canvas.ax, width=edge_widths, edge_color=edge_colors, alpha=0.9
        )
        label_kwargs = {"font_size": 8}
        if self.network_font_name:
            label_kwargs["font_family"] = self.network_font_name
        nx.draw_networkx_labels(graph, pos, ax=self.network_canvas.ax, **label_kwargs)
        self.network_canvas.ax.set_title("네트워크")
        self.network_canvas.ax.axis("off")
        self.network_canvas.draw()

    def on_network_press(self, event):
        if not self.chk_drag_mode.isChecked():
            return
        if self.graph_view is None or event.inaxes != self.network_canvas.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        closest = None
        closest_dist = None
        for node, (x_pos, y_pos) in self.network_pos.items():
            dist = (x_pos - event.xdata) ** 2 + (y_pos - event.ydata) ** 2
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest = node
        if closest_dist is not None and closest_dist < 0.02:
            self.network_drag_node = closest

    def on_network_motion(self, event):
        if not self.chk_drag_mode.isChecked():
            return
        if self.network_drag_node is None or event.inaxes != self.network_canvas.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.network_pos[self.network_drag_node] = (event.xdata, event.ydata)
        self.draw_network(self.graph_view)

    def on_network_release(self, event):
        if not self.chk_drag_mode.isChecked():
            return
        self.network_drag_node = None

    def populate_network_tables(self, edges_df, nodes_df):
        self.tbl_edges.setRowCount(len(edges_df))
        for row_idx, (_, row) in enumerate(edges_df.iterrows()):
            self.tbl_edges.setItem(row_idx, 0, QTableWidgetItem(str(row["source"])))
            self.tbl_edges.setItem(row_idx, 1, QTableWidgetItem(str(row["target"])))
            self.tbl_edges.setItem(row_idx, 2, QTableWidgetItem(f"{row['weight']:.2f}"))

        self.tbl_nodes.setRowCount(len(nodes_df))
        for row_idx, (_, row) in enumerate(nodes_df.iterrows()):
            self.tbl_nodes.setItem(row_idx, 0, QTableWidgetItem(str(row["node"])))
            self.tbl_nodes.setItem(row_idx, 1, QTableWidgetItem(str(row["degree"])))

    def run_sentiment(self):
        if self.df_clean is None:
            return
        lang_mode = self.get_sentiment_language_mode()
        self.update_sentiment_lexicon(include_english=lang_mode != "ko")

        df = self.apply_monthly_sampling(self.df_clean)
        text_col = self.get_analysis_text_column(df)
        records = []
        total_token_count = 0
        for _, row in df.iterrows():
            text = row.get(text_col, "")
            prev_final_score = None
            for sentence in self.split_sentiment_sentences(text):
                sentence_lang = lang_mode
                apply_rules = self.chk_rule_master.isChecked()
                use_neg_scope = apply_rules and self.chk_rule_neg_scope.isChecked()
                use_contrast = apply_rules and self.chk_rule_contrast.isChecked()
                use_emoticon = apply_rules and self.chk_rule_emoticon.isChecked()
                use_profanity = apply_rules and self.chk_rule_profanity.isChecked()
                use_prev_negative = apply_rules and self.chk_rule_prev_negative.isChecked()

                if use_contrast:
                    clauses = self.split_contrast_clauses(sentence)
                else:
                    clauses = [sentence]

                clause_scores = []
                matched_tokens = []
                reasons = []
                for idx, clause in enumerate(clauses):
                    clause_lang = sentence_lang
                    if clause_lang == "unknown":
                        clause_lang = "ko"
                    clause_score = 0
                    clause_matched = []
                    clause_reasons = []
                    if clause_lang == "ko":
                        tokens = self.tokenize_sentiment_korean(clause)
                        total_token_count += len(tokens)
                        matches = self.match_sentiment_tokens(tokens, self.senti_dict, self.senti_max_n)
                        clause_score, clause_matched, clause_reasons = self.calculate_sentence_score(
                            clause,
                            tokens,
                            matches,
                            use_neg_scope,
                            self.senti_dict,
                            NEGATION_TOKENS_KO,
                        )
                    elif clause_lang == "en":
                        tokens = self.tokenize_sentiment_english(clause)
                        total_token_count += len(tokens)
                        matches = self.match_sentiment_tokens(
                            tokens,
                            self.senti_dict_en,
                            self.senti_max_n_en,
                        )
                        clause_score, clause_matched, clause_reasons = self.calculate_sentence_score(
                            clause,
                            tokens,
                            matches,
                            use_neg_scope,
                            self.senti_dict_en,
                            NEGATION_TOKENS_EN,
                        )
                    else:
                        ko_tokens = self.tokenize_sentiment_korean(clause)
                        en_tokens = self.tokenize_sentiment_english(clause)
                        total_token_count += len(ko_tokens) + len(en_tokens)
                        ko_matches = self.match_sentiment_tokens(
                            ko_tokens,
                            self.senti_dict,
                            self.senti_max_n,
                        )
                        en_matches = self.match_sentiment_tokens(
                            en_tokens,
                            self.senti_dict_en,
                            self.senti_max_n_en,
                        )
                        ko_score, ko_matched, ko_reasons = self.calculate_sentence_score(
                            clause,
                            ko_tokens,
                            ko_matches,
                            use_neg_scope,
                            self.senti_dict,
                            NEGATION_TOKENS_KO,
                        )
                        en_score, en_matched, en_reasons = self.calculate_sentence_score(
                            clause,
                            en_tokens,
                            en_matches,
                            use_neg_scope,
                            self.senti_dict_en,
                            NEGATION_TOKENS_EN,
                        )
                        clause_score = ko_score + en_score
                        clause_matched = ko_matched + en_matched
                        clause_reasons = ko_reasons + en_reasons
                    weight = 1.0 if idx == 0 else 1.5
                    if len(clauses) > 1 and idx > 0:
                        reasons.append("대조 접속어")
                    clause_scores.append(clause_score * weight)
                    matched_tokens.extend(clause_matched)
                    reasons.extend(clause_reasons)

                raw_score = sum(clause_scores) if clause_scores else 0

                if use_emoticon and EMOTICON_PATTERN.search(sentence) and POSITIVE_PATTERN.search(sentence):
                    if raw_score <= 0:
                        raw_score = max(raw_score + 1, 1)
                        reasons.append("이모티콘+긍정 보정")
                if use_profanity and PROFANITY_PATTERN.search(sentence) and POSITIVE_PATTERN.search(sentence):
                    raw_score = max(raw_score + 1, 1)
                    reasons.append("비속어+긍정 강화")

                if use_prev_negative and prev_final_score is not None:
                    if prev_final_score <= -1 and 0 <= raw_score <= 1:
                        raw_score -= 1
                        reasons.append("이전 문장 부정 전파")

                score = self.bin_sentiment_score(raw_score)
                label = self.sentiment_score_label(score)
                topic = self.match_topic(sentence)
                adjust_reason = "; ".join(dict.fromkeys(reasons))
                records.append(
                    {
                        "date": row.get("date"),
                        "page_type": row.get("page_type"),
                        "sentence": sentence,
                        "score": score,
                        "label": label,
                        "topic": topic,
                        "matched_words": ", ".join(matched_tokens),
                        "adjusted_score": score,
                        "adjust_reason": adjust_reason,
                    }
                )
                prev_final_score = score

        if total_token_count == 0:
            self.warn_no_tokens("감성분석")
            self.sentiment_records_df = pd.DataFrame()
            self.sentiment_summary_df = None
            self.tbl_sent_records.setRowCount(0)
            self.sent_canvas.ax.clear()
            self.sent_canvas.draw()
            self.sent_trend_canvas.ax.clear()
            self.sent_trend_canvas.draw()
            self.clear_sentiment_topic_charts()
            self.txt_voc.clear()
            return

        self.sentiment_records_df = pd.DataFrame(records)
        self.update_sentiment_view()

    def update_sentiment_view(self):
        if self.sentiment_records_df is None or self.sentiment_records_df.empty:
            return
        df = self.filter_sentiment_by_period(self.sentiment_records_df.copy())
        if df.empty:
            self.tbl_sent_records.setRowCount(0)
            self.sent_canvas.ax.clear()
            self.sent_canvas.draw()
            self.sent_trend_canvas.ax.clear()
            self.sent_trend_canvas.draw()
            self.clear_sentiment_topic_charts()
            self.txt_voc.clear()
            self.statusBar().showMessage("선택한 기간에 감성 데이터가 없습니다.")
            return
        self.update_sentiment_range_label(df)
        mode = self.cb_sent_mode.currentText()
        topic_filter = self.cb_brand_filter.currentText()
        page_filter = self.cb_sent_page_type.currentText()
        if mode == "사전별 감성" and not self.ensure_topic_dictionary():
            self.cb_sent_mode.blockSignals(True)
            self.cb_sent_mode.setCurrentText("전체 감성")
            self.cb_sent_mode.blockSignals(False)
            mode = "전체 감성"
            topic_filter = "전체"
        self.cb_brand_filter.setEnabled(mode == "사전별 감성")
        if mode == "사전별 감성" and topic_filter != "전체":
            df = df[df["topic"] == topic_filter]
        if page_filter != "전체":
            df = df[df["page_type"] == page_filter]
        if df.empty:
            self.tbl_sent_records.setRowCount(0)
            self.sent_canvas.ax.clear()
            self.sent_canvas.draw()
            self.sent_trend_canvas.ax.clear()
            self.sent_trend_canvas.draw()
            self.clear_sentiment_topic_charts()
            self.txt_voc.clear()
            self.statusBar().showMessage("선택한 필터에 감성 데이터가 없습니다.")
            return

        view = self.cb_sent_view.currentText()
        if view == "월별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m")
        elif view == "주별":
            df["bucket"] = df["date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        elif view == "일별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m-%d")
        else:
            df["bucket"] = df["date"].dt.strftime("%Y")

        score_order = [-2, -1, 0, 1, 2]
        summary = (
            df.groupby(["bucket", "score", "label", "topic"])
            .size()
            .reset_index(name="count")
        )
        self.sentiment_summary_df = summary

        if mode == "사전별 감성":
            bucket_summary = (
                df.groupby(["bucket", "topic", "score"])
                .size()
                .reset_index(name="count")
            )
            pivot = bucket_summary.pivot_table(
                index=["bucket", "topic"], columns="score", values="count", fill_value=0
            ).reindex(columns=score_order, fill_value=0)
            pivot["total"] = pivot.sum(axis=1)
            pivot = pivot.reset_index()
            pivot = pivot.rename(columns={score: self.sentiment_bucket_label(score) for score in score_order})
            self.populate_sentiment_bucket_table(pivot)
            chart_summary = df.groupby(["score", "topic"]).size().reset_index(name="count")
        else:
            self.populate_sentiment_table(df)
            chart_summary = df.groupby(["score", "label"]).size().reset_index(name="count")

        self.plot_sentiment_chart(chart_summary)
        self.plot_sentiment_topic_stacks(df)
        self.plot_sentiment_trend(df)
        self.update_voc_summary(df)

    def update_sentiment_range_label(self, df):
        if df is None or df.empty:
            self.lbl_sent_period_range.setText("")
            return
        min_date = df["date"].min()
        max_date = df["date"].max()
        view = self.cb_sent_view.currentText()
        if view == "월별":
            label = f"{min_date.strftime('%Y년 %m월')} ~ {max_date.strftime('%Y년 %m월')}"
        elif view == "주별":
            min_week = (min_date.day - 1) // 7 + 1
            max_week = (max_date.day - 1) // 7 + 1
            label = (
                f"{min_date.strftime('%Y년 %m월')} {min_week}주 ~ "
                f"{max_date.strftime('%Y년 %m월')} {max_week}주"
            )
        elif view == "일별":
            min_week = (min_date.day - 1) // 7 + 1
            max_week = (max_date.day - 1) // 7 + 1
            label = (
                f"{min_date.strftime('%Y년 %m월')} {min_week}주 {min_date.day}일 ~ "
                f"{max_date.strftime('%Y년 %m월')} {max_week}주 {max_date.day}일"
            )
        else:
            label = f"{min_date.strftime('%Y년')} ~ {max_date.strftime('%Y년')}"
        self.lbl_sent_period_range.setText(label)

    def populate_sentiment_table(self, df):
        show_records = df.head(200)
        self.tbl_sent_records.setRowCount(len(show_records))
        self.tbl_sent_records.setColumnCount(6)
        self.tbl_sent_records.setHorizontalHeaderLabels([
            "date",
            "page_type",
            "sentence",
            "score",
            "label",
            "topic",
        ])
        for row_idx, (idx, row) in enumerate(show_records.iterrows()):
            date_item = QTableWidgetItem(self.format_date(row["date"]))
            date_item.setData(Qt.UserRole, idx)
            self.tbl_sent_records.setItem(row_idx, 0, date_item)
            self.tbl_sent_records.setItem(row_idx, 1, QTableWidgetItem(str(row["page_type"])))
            self.tbl_sent_records.setItem(row_idx, 2, QTableWidgetItem(str(row["sentence"])))
            self.tbl_sent_records.setItem(row_idx, 3, QTableWidgetItem(str(row["score"])))
            self.tbl_sent_records.setItem(row_idx, 4, QTableWidgetItem(str(row["label"])))
            self.tbl_sent_records.setItem(row_idx, 5, QTableWidgetItem(str(row["topic"])))

    def populate_sentiment_summary_table(self, summary_df):
        self.tbl_sent_records.setRowCount(len(summary_df))
        self.tbl_sent_records.setColumnCount(6)
        self.tbl_sent_records.setHorizontalHeaderLabels([
            "topic",
            "page_type",
            "avg_score",
            "pos_pct",
            "neg_pct",
            "count",
        ])
        for row_idx, (_, row) in enumerate(summary_df.iterrows()):
            self.tbl_sent_records.setItem(row_idx, 0, QTableWidgetItem(str(row["topic"])))
            self.tbl_sent_records.setItem(row_idx, 1, QTableWidgetItem(str(row["page_type"])))
            self.tbl_sent_records.setItem(row_idx, 2, QTableWidgetItem(f"{row['avg_score']:.2f}"))
            self.tbl_sent_records.setItem(row_idx, 3, QTableWidgetItem(f"{row['pos_pct']:.2f}"))
            self.tbl_sent_records.setItem(row_idx, 4, QTableWidgetItem(f"{row['neg_pct']:.2f}"))
            self.tbl_sent_records.setItem(row_idx, 5, QTableWidgetItem(str(int(row["count"]))))

    def populate_sentiment_bucket_table(self, summary_df):
        headers = list(summary_df.columns)
        self.tbl_sent_records.setRowCount(len(summary_df))
        self.tbl_sent_records.setColumnCount(len(headers))
        self.tbl_sent_records.setHorizontalHeaderLabels(headers)
        for row_idx, (_, row) in enumerate(summary_df.iterrows()):
            for col_idx, col in enumerate(headers):
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                self.tbl_sent_records.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def update_voc_summary(self, df):
        if df.empty:
            self.txt_voc.clear()
            return
        voc_lines = []
        for topic, sub in df.groupby("topic"):
            voc_lines.append(f"[{topic}]")
            positives = sub[sub["score"] > 0].head(3)
            negatives = sub[sub["score"] < 0].head(3)
            if not positives.empty:
                voc_lines.append("긍정")
                for _, row in positives.iterrows():
                    reason = row.get("adjust_reason", "")
                    extra = f"{row.get('matched_words', '')}"
                    if reason:
                        extra = f"{extra} | {reason}" if extra else reason
                    voc_lines.append(f"- {row['sentence']} ({extra})")
            if not negatives.empty:
                voc_lines.append("부정")
                for _, row in negatives.iterrows():
                    reason = row.get("adjust_reason", "")
                    extra = f"{row.get('matched_words', '')}"
                    if reason:
                        extra = f"{extra} | {reason}" if extra else reason
                    voc_lines.append(f"- {row['sentence']} ({extra})")
            voc_lines.append("")
        self.txt_voc.setPlainText("\n".join(voc_lines))

    def plot_sentiment_chart(self, summary):
        metric = self.cb_sent_metric.currentText()
        score_order = [-2, -1, 0, 1, 2]
        labels = [self.sentiment_bucket_label(score) for score in score_order]
        if metric == "%":
            note_text = "※ 라벨은 비중"
            label_mode = "percent"
        else:
            note_text = "※ 라벨은 건수"
            label_mode = "count"
        if "topic" in summary.columns:
            pivot_counts = summary.pivot_table(
                index="score", columns="topic", values="count", fill_value=0
            ).reindex(score_order, fill_value=0)
            if metric == "%":
                pivot = pivot_counts.div(pivot_counts.sum(axis=1).replace(0, 1), axis=0) * 100
                ylabel = "%"
            else:
                pivot = pivot_counts
                ylabel = "count"
            series = [(str(col), pivot[col].tolist()) for col in pivot_counts.columns]
            count_series = [pivot_counts[col].tolist() for col in pivot_counts.columns]
            self.sent_canvas.plot_stacked_bar(
                labels,
                series,
                "감성 분포",
                ylabel,
                count_series=count_series,
                note_text=note_text,
                label_mode=label_mode,
            )
        else:
            counts = {score: 0 for score in score_order}
            for _, row in summary.iterrows():
                counts[row["score"]] = int(row["count"])
            count_values = [counts[score] for score in score_order]
            if metric == "%":
                total = sum(count_values) or 1
                values = [round(value / total * 100, 2) for value in count_values]
                ylabel = "%"
            else:
                values = count_values
                ylabel = "count"
            self.sent_canvas.ax.clear()
            bar_colors = [SENTIMENT_COLORS[score] for score in score_order]
            if labels:
                bars = self.sent_canvas.ax.bar(labels, values, color=bar_colors)
                self.sent_canvas.ax.set_xticks(range(len(labels)))
                self.sent_canvas.ax.set_xticklabels(labels, rotation=45, ha="right")
                total = sum(count_values) or 1
                percents = [(count / total) * 100 for count in count_values]
                self.sent_canvas.annotate_bars(bars, count_values, percents, label_mode=label_mode)
            self.sent_canvas.ax.set_title("감성 분포")
            self.sent_canvas.ax.set_ylabel(ylabel)
            self.sent_canvas.ax.grid(axis="y", alpha=0.3)
            self.sent_canvas.add_bar_label_note(note_text)
            self.sent_canvas.figure.tight_layout()
            self.sent_canvas.draw()
        self.chart_images["sentiment"] = self.save_chart(self.sent_canvas, "sentiment")

    def clear_sentiment_topic_charts(self):
        for idx in reversed(range(self.sent_topic_charts_layout.count())):
            item = self.sent_topic_charts_layout.takeAt(idx)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def plot_sentiment_topic_stacks(self, df):
        self.clear_sentiment_topic_charts()
        if df is None or df.empty:
            return
        score_order = [-2, -1, 0, 1, 2]
        labels = [self.sentiment_bucket_label(score) for score in score_order]
        metric = self.cb_sent_metric.currentText()
        summary = df.groupby(["topic", "score"]).size().reset_index(name="count")
        pivot_counts = summary.pivot_table(index="topic", columns="score", values="count", fill_value=0)
        pivot_counts = pivot_counts.reindex(columns=score_order, fill_value=0)
        if metric == "%":
            pivot = pivot_counts.div(pivot_counts.sum(axis=1).replace(0, 1), axis=0) * 100
            ylabel = "%"
            label_mode = "percent"
        else:
            pivot = pivot_counts
            ylabel = "count"
            label_mode = "count"
        topics = list(pivot_counts.index)
        series = [
            (label, pivot[score].tolist(), SENTIMENT_COLORS.get(score, "#999999"))
            for label, score in zip(labels, score_order)
        ]
        self.sent_topic_canvas.ax.clear()
        x_positions = list(range(len(topics)))
        bottoms = [0] * len(topics)
        totals = pivot_counts.sum(axis=1).replace(0, 1).tolist()
        note_text = "※ 라벨은 비중" if metric == "%" else "※ 라벨은 건수"
        for score_idx, (label, values, color) in enumerate(series):
            bars = self.sent_topic_canvas.ax.bar(
                x_positions, values, bottom=bottoms, label=label, color=color
            )
            counts = pivot_counts[score_order[score_idx]].tolist()
            percents = [(count / total) * 100 for count, total in zip(counts, totals)]
            for bar, bottom, count, percent in zip(bars, bottoms, counts, percents):
                if count <= 0:
                    continue
                if label_mode == "percent":
                    label_text = self.sent_topic_canvas.format_percent_label(percent)
                else:
                    label_text = self.sent_topic_canvas.format_count_label(count)
                self.sent_topic_canvas.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + bar.get_height() / 2,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if bar.get_height() > 0 else "#333333",
                )
            bottoms = [b + v for b, v in zip(bottoms, values)]
        self.sent_topic_canvas.ax.set_xticks(x_positions)
        self.sent_topic_canvas.ax.set_xticklabels(topics, rotation=45, ha="right")
        self.sent_topic_canvas.ax.set_ylabel(ylabel)
        self.sent_topic_canvas.ax.set_title("사전 대표단어 감성 스택")
        self.sent_topic_canvas.ax.legend(loc="upper right")
        self.sent_topic_canvas.add_bar_label_note(note_text)
        self.sent_topic_canvas.figure.tight_layout()
        self.sent_topic_canvas.draw()
        self.sent_topic_charts_layout.addWidget(self.sent_topic_canvas)

    def plot_sentiment_trend(self, df):
        if df is None or df.empty:
            self.sent_trend_canvas.ax.clear()
            self.sent_trend_canvas.draw()
            return
        view = self.cb_sent_view.currentText()
        if view == "월별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m")
        elif view == "주별":
            df["bucket"] = df["date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        elif view == "일별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m-%d")
        else:
            df["bucket"] = df["date"].dt.strftime("%Y")
        score_order = [-2, -1, 0, 1, 2]
        summary = (
            df.groupby(["bucket", "score"])
            .size()
            .reset_index(name="count")
        )
        pivot = summary.pivot_table(index="bucket", columns="score", values="count", fill_value=0)
        pivot = pivot.reindex(columns=score_order, fill_value=0)
        pivot = pivot.sort_index()
        if self.cb_sent_metric.currentText() == "%":
            pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
            ylabel = "%"
        else:
            ylabel = "count"
        labels = list(pivot.index)
        series = []
        colors_map = {}
        for score in score_order:
            label = self.sentiment_bucket_label(score)
            series.append((label, pivot[score].tolist()))
            colors_map[label] = SENTIMENT_COLORS.get(score)
        self.sent_trend_canvas.plot_multi_line(
            labels,
            series,
            "감성 추이",
            ylabel,
            colors_map=colors_map,
        )
        self.chart_images["sentiment_trend"] = self.save_chart(self.sent_trend_canvas, "sentiment_trend")

    def update_network_recommendation(self, doc_count: int, token_count: int):
        if doc_count <= 100:
            min_node = 2
            min_edge = 2
        elif doc_count <= 500:
            min_node = 5
            min_edge = 3
        else:
            min_node = 10
            min_edge = 5
        max_nodes = min(200, max(30, int(token_count * 0.3)))
        self.lbl_network_reco.setText(
            f"데이터 {doc_count}건/토큰 {token_count}개 기준 권장: "
            f"노드 최소 {min_node}, 엣지 최소 {min_edge}, 최대 노드 {max_nodes}"
        )

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if folder:
            self.lbl_out_dir.setText(folder)

    def export_excel(self):
        if not self.lbl_out_dir.text() or self.lbl_out_dir.text() == "저장 폴더 미선택":
            self.statusBar().showMessage("저장 폴더를 선택해주세요.")
            return
        filename = self.le_filename.text().strip()
        if not filename:
            self.statusBar().showMessage("파일명을 입력해주세요.")
            return
        path = os.path.join(self.lbl_out_dir.text(), filename)
        items = {
            self.list_export_items.item(i).text()
            for i in range(self.list_export_items.count())
            if self.list_export_items.item(i).checkState() == Qt.Checked
        }

        writer_kwargs = {"engine": "openpyxl"}
        if os.path.exists(path):
            writer_kwargs.update({"mode": "a", "if_sheet_exists": "replace"})
        with pd.ExcelWriter(path, **writer_kwargs) as writer:
            if "clean_data" in items and self.df_clean is not None:
                self.df_clean.to_excel(writer, sheet_name="clean_data", index=False)
            if "buzz_summary" in items and self.buzz_df is not None:
                self.buzz_df.to_excel(writer, sheet_name="buzz_summary", index=False)
            if "word_freq_topN" in items and self.word_freq_df is not None:
                self.word_freq_df.to_excel(writer, sheet_name="word_freq_topN", index=False)
            if "network_nodes" in items and self.nodes_df is not None:
                self.nodes_df.to_excel(writer, sheet_name="network_nodes", index=False)
            if "network_edges" in items and self.edges_df is not None:
                self.edges_df.to_excel(writer, sheet_name="network_edges", index=False)
            if "sentiment_records" in items and self.sentiment_records_df is not None:
                self.sentiment_records_df.to_excel(writer, sheet_name="sentiment_records", index=False)
            if "sentiment_summary" in items and self.sentiment_summary_df is not None:
                self.sentiment_summary_df.to_excel(writer, sheet_name="sentiment_summary", index=False)
            if "charts" in items:
                pd.DataFrame({"chart": list(self.chart_images.keys())}).to_excel(
                    writer, sheet_name="charts", index=False
                )
                chart_sheet = writer.book["charts"]
                row = 3
                for _, image_path in self.chart_images.items():
                    if not os.path.exists(image_path):
                        continue
                    img = XLImage(image_path)
                    img.anchor = f"A{row}"
                    chart_sheet.add_image(img)
                    row += 20

        self.txt_export_log.append(f"저장 완료: {path}")
        return path

    def save_chart(self, canvas, name):
        path = os.path.join(os.getcwd(), f"{name}.png")
        canvas.figure.savefig(path, dpi=150)
        return path

    def match_brand(self, text: str):
        if not self.brand_map or not isinstance(text, str):
            return "전체"
        lowered = text.lower()
        for brand, keywords in self.brand_map.items():
            for keyword in keywords:
                if keyword.lower() in lowered:
                    return brand
        return "전체"


def main():
    app = QApplication(sys.argv)

    font_path = resolve_font_path()
    if font_path:
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QFont(families[0], 10))
    configure_matplotlib_font(font_path)

    window = TextMiningApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
