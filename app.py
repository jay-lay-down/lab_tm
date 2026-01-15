import itertools
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib import font_manager as fm
import networkx as nx
import numpy as np
import pandas as pd
import requests
from kiwipiepy import Kiwi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
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

matplotlib.use("Qt5Agg")

KNU_DICT_URL = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json"
DEFAULT_RESOURCE_DIR = Path(r"C:\Users\70089004\text_file")
DEFAULT_FONT_NAME = "Pretendard-Medium.otf"
DEFAULT_SENTI_NAME = "SentiWord_Dict.txt"
DEFAULT_NETWORK_FONT_NAME = "malgun.ttf"
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
NEGATION_TOKENS = {"안", "못", "별로", "전혀", "아니", "없", "않"}
CONTRAST_TOKENS = ["하지만", "근데", "그런데", "그러나"]


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


def parse_sentiment_entries(entries):
    senti_dict = {}
    for entry in entries:
        if isinstance(entry, dict):
            root = entry.get("word_root", entry.get("word"))
            score = int(entry.get("polarity", 0))
            if root:
                senti_dict[root] = score
            continue
        if isinstance(entry, str):
            parts = re.split(r"[\t,]", entry.strip())
            if len(parts) >= 2:
                root = parts[0].strip()
                if root:
                    try:
                        score = int(parts[1].strip())
                    except ValueError:
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

    def plot_bar(self, labels, values, title, ylabel):
        self.ax.clear()
        if labels:
            self.ax.bar(labels, values, color="#4b77be")
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
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

    def plot_multi_bar(self, labels, series, title, ylabel):
        self.ax.clear()
        if labels and series:
            total = len(series)
            width = 0.8 / total
            x_positions = list(range(len(labels)))
            for idx, (name, values) in enumerate(series):
                offset = (idx - (total - 1) / 2) * width
                positions = [x + offset for x in x_positions]
                self.ax.bar(positions, values, width=width, label=name)
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
        self.figure.tight_layout()
        self.draw()

    def plot_stacked_bar(self, labels, series, title, ylabel):
        self.ax.clear()
        if labels and series:
            x_positions = list(range(len(labels)))
            bottoms = [0] * len(labels)
            for name, values in series:
                self.ax.bar(x_positions, values, bottom=bottoms, label=name)
                bottoms = [b + v for b, v in zip(bottoms, values)]
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(axis="y", alpha=0.3)
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

        self.senti_dict = None
        self.senti_max_n = 1
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

    def _build_tab_data_load(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizes([200, 600])
        main_layout.addWidget(splitter)

        top = QWidget()
        top.setMinimumHeight(160)
        top.setMaximumHeight(220)
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

        top_layout.addWidget(self.group_page_type_filter, 2, 0, 2, 2)
        top_layout.addWidget(self.group_keyword_filter, 2, 2, 2, 3)

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
        self.cb_granularity.addItems(["연도", "월"])
        self.cb_buzz_period_unit = self.cb_granularity
        self.cb_buzz_period_value = QComboBox()
        self.cb_buzz_period_unit.currentIndexChanged.connect(
            lambda: self.populate_period_values(
                self.cb_buzz_period_unit, self.cb_buzz_period_value, self.df_clean
            )
        )
        self.chk_split_by_page_type = QCheckBox("page_type 분리")
        self.cb_page_type_filter = QComboBox()
        self.cb_page_type_filter.addItem("전체")
        self.cb_buzz_metric = QComboBox()
        self.cb_buzz_metric.addItems(["n", "%"])
        self.btn_refresh_buzz = QPushButton("버즈 계산")
        self.btn_refresh_buzz.clicked.connect(self.build_buzz)

        top_layout.addWidget(QLabel("기간 단위"), 0, 0)
        top_layout.addWidget(self.cb_granularity, 0, 1)
        top_layout.addWidget(QLabel("기간 선택"), 0, 2)
        top_layout.addWidget(self.cb_buzz_period_value, 0, 3)
        top_layout.addWidget(self.chk_split_by_page_type, 0, 4)
        top_layout.addWidget(QLabel("page_type"), 0, 5)
        top_layout.addWidget(self.cb_page_type_filter, 0, 6)
        top_layout.addWidget(QLabel("지표"), 0, 7)
        top_layout.addWidget(self.cb_buzz_metric, 0, 8)
        top_layout.addWidget(self.btn_refresh_buzz, 0, 9)

        layout.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([750, 450])
        self.buzz_canvas = ChartCanvas()
        self.tbl_buzz = QTableWidget()
        self.tbl_buzz.setColumnCount(3)
        self.tbl_buzz.setHorizontalHeaderLabels(["date", "metric", "value"])
        self.tbl_buzz.horizontalHeader().setStretchLastSection(True)

        splitter.addWidget(self.buzz_canvas)
        splitter.addWidget(self.tbl_buzz)
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
        self.cb_wc_topn.addItems(["30", "50", "100", "200", "500", "1000", "2000"])
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

        splitter.addWidget(self.lbl_wc_view)
        splitter.addWidget(self.tbl_wc_topn)
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
        self.btn_build_graph = QPushButton("그래프 생성(전체)")
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
        top_layout.addWidget(self.btn_build_graph, 0, 4)
        top_layout.addWidget(self.cb_mode, 0, 5)
        top_layout.addWidget(self.lbl_min_node, 0, 6)
        top_layout.addWidget(self.sb_min_node_count, 0, 7)
        top_layout.addWidget(self.lbl_min_edge, 0, 8)
        top_layout.addWidget(self.sb_min_edge_weight, 0, 9)
        top_layout.addWidget(self.lbl_max_nodes, 0, 10)
        top_layout.addWidget(self.sb_max_nodes, 0, 11)
        top_layout.addWidget(self.le_node_search, 1, 0, 1, 2)
        top_layout.addWidget(self.btn_add_seed, 1, 2)
        top_layout.addWidget(self.cb_hop_depth, 1, 3)
        top_layout.addWidget(self.btn_apply_hop, 1, 4)
        top_layout.addWidget(self.btn_reset_view, 1, 5)
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
        bottom.addWidget(self.network_canvas)
        bottom.addWidget(self.network_tabs)
        splitter.addWidget(bottom)

        self.tabs.addTab(tab, "네트워크")
        self.toggle_network_advanced()

    def _build_tab_sentiment(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QWidget()
        top.setFixedHeight(190)
        top_layout = QVBoxLayout(top)
        filter_row = QWidget()
        filter_layout = QGridLayout(filter_row)
        self.cb_sent_period_value = QComboBox()
        self.cb_sent_period_value.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sent_mode = QComboBox()
        self.cb_sent_mode.addItems(["전체 감성", "사전별 감성"])
        self.cb_sent_mode.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sent_view = QComboBox()
        self.cb_sent_view.addItems(["연도별", "월별"])
        self.cb_sent_view.currentIndexChanged.connect(
            lambda: (self.populate_sentiment_period_values(), self.update_sentiment_view())
        )
        self.cb_sent_metric = QComboBox()
        self.cb_sent_metric.addItems(["count", "%"])
        self.cb_sent_metric.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sentence_split = QComboBox()
        self.cb_sentence_split.addItems(["기본", "강함(쉼표 포함)"])
        self.cb_brand_filter = QComboBox()
        self.cb_brand_filter.addItem("전체")
        self.cb_brand_filter.currentIndexChanged.connect(self.update_sentiment_view)
        self.btn_run_sentiment = QPushButton("감성분석 실행")
        self.btn_run_sentiment.clicked.connect(self.run_sentiment)
        self.chk_monthly_sample_sent = self.create_monthly_sampling_checkbox()

        filter_layout.addWidget(QLabel("기간 선택"), 0, 0)
        filter_layout.addWidget(self.cb_sent_period_value, 0, 1)
        filter_layout.addWidget(QLabel("모드"), 0, 2)
        filter_layout.addWidget(self.cb_sent_mode, 0, 3)
        filter_layout.addWidget(QLabel("보기"), 0, 4)
        filter_layout.addWidget(self.cb_sent_view, 0, 5)
        filter_layout.addWidget(QLabel("지표"), 0, 6)
        filter_layout.addWidget(self.cb_sent_metric, 0, 7)
        filter_layout.addWidget(self.chk_monthly_sample_sent, 0, 8)
        filter_layout.addWidget(QLabel("문장 분리"), 1, 0)
        filter_layout.addWidget(self.cb_sentence_split, 1, 1)
        filter_layout.addWidget(self.btn_run_sentiment, 1, 2)
        filter_layout.addWidget(QLabel("토픽"), 1, 3)
        filter_layout.addWidget(self.cb_brand_filter, 1, 4, 1, 2)
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
        self.sent_chart_tabs = QTabWidget()
        self.sent_canvas = ChartCanvas()
        self.sent_topic_charts_container = QWidget()
        self.sent_topic_charts_layout = QVBoxLayout(self.sent_topic_charts_container)
        self.sent_topic_charts_layout.setContentsMargins(8, 8, 8, 8)
        self.sent_topic_charts_layout.setSpacing(12)
        self.sent_topic_scroll = QScrollArea()
        self.sent_topic_scroll.setWidgetResizable(True)
        self.sent_topic_scroll.setWidget(self.sent_topic_charts_container)
        self.sent_trend_canvas = ChartCanvas()
        self.sent_chart_tabs.addTab(self.sent_canvas, "감성 분포")
        self.sent_chart_tabs.addTab(self.sent_topic_scroll, "사전별 감성 스택")
        self.sent_chart_tabs.addTab(self.sent_trend_canvas, "감성 추이")
        splitter.addWidget(self.tbl_sent_records)
        splitter.addWidget(self.sent_chart_tabs)

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
        for idx in range(1, 7):
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
        else:
            values = sorted({val.strftime("%Y-%m") for val in dates})
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
        return df[dates.dt.strftime("%Y-%m") == value]

    def filter_sentiment_by_period(self, df: pd.DataFrame):
        if df is None or df.empty or "date" not in df.columns:
            return df
        value = self.cb_sent_period_value.currentText()
        if value in {PERIOD_ALL_LABEL, "전체"}:
            return df
        view = self.cb_sent_view.currentText()
        if view == "월별":
            return df[df["date"].dt.strftime("%Y-%m") == value]
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
        self.cb_page_type_filter.blockSignals(True)
        self.cb_page_type_filter.clear()
        self.cb_page_type_filter.addItem("전체")
        if self.df_raw is None:
            self.cb_page_type_filter.blockSignals(False)
            return
        mapping = self.map_columns(self.df_raw)
        page_col = mapping.get("page_type")
        if page_col is None:
            self.cb_page_type_filter.blockSignals(False)
            return
        unique_vals = sorted({str(val) for val in self.df_raw[page_col].dropna().unique()})
        for val in unique_vals:
            item = QListWidgetItem(val)
            item.setCheckState(Qt.Unchecked)
            self.list_page_type.addItem(item)
            self.cb_page_type_filter.addItem(val)
        self.cb_page_type_filter.blockSignals(False)

    def select_all_page_types(self):
        for idx in range(self.list_page_type.count()):
            self.list_page_type.item(idx).setCheckState(Qt.Checked)

    def clear_all_page_types(self):
        for idx in range(self.list_page_type.count()):
            self.list_page_type.item(idx).setCheckState(Qt.Unchecked)

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
        df = self.df_raw.copy()
        df_mapped = pd.DataFrame()
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

        self.df_clean = df_mapped[["date", "page_type", "full_text"]]
        self.lbl_rows.setText(f"원본 {len(self.df_raw)} → 현재 {len(self.df_clean)}")
        self.update_preview(self.df_clean)
        self.update_gate_state()

    def update_preview(self, df):
        preview = df.head(200)
        self.tbl_preview.setRowCount(len(preview))
        for row_idx, (_, row) in enumerate(preview.iterrows()):
            self.tbl_preview.setItem(row_idx, 0, QTableWidgetItem(self.format_date(row.get("date"))))
            self.tbl_preview.setItem(row_idx, 1, QTableWidgetItem(str(row.get("page_type", ""))))
            self.tbl_preview.setItem(row_idx, 2, QTableWidgetItem(str(row.get("full_text", ""))))

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

    def update_sentiment_lexicon(self):
        if self.senti_dict is None:
            self.senti_dict = load_knu_dictionary(self)
        self.senti_max_n = 1
        for word in self.senti_dict.keys():
            token_len = max(1, len(str(word).split()))
            if token_len > self.senti_max_n:
                self.senti_max_n = token_len

    def match_sentiment_tokens(self, tokens):
        if not tokens or not self.senti_dict:
            return []
        used = [False] * len(tokens)
        matched = []
        for n in range(self.senti_max_n, 0, -1):
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
                if cand1 in self.senti_dict:
                    hit = cand1
                elif cand2 in self.senti_dict:
                    hit = cand2
                if hit:
                    matched.append({"token": hit, "start": idx, "end": idx + n})
                    for pos in range(idx, idx + n):
                        used[pos] = True
                    idx += n
                else:
                    idx += 1
        return matched

    def apply_negation_scope(self, tokens, matches):
        adjusted_scores = []
        negated_tokens = set()
        for match in matches:
            token = match["token"]
            start = match["start"]
            score = self.senti_dict.get(token, 0)
            scope_start = max(0, start - 3)
            scope_tokens = tokens[scope_start:start]
            if any(scope in NEGATION_TOKENS for scope in scope_tokens):
                score = -score
                negated_tokens.add(token)
            adjusted_scores.append(score)
        return adjusted_scores, negated_tokens

    def split_contrast_clauses(self, sentence: str):
        for token in CONTRAST_TOKENS:
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

    def calculate_sentence_score(self, sentence: str, tokens, matches, apply_neg_scope: bool):
        matched_tokens = [match["token"] for match in matches]
        reasons = []
        if not matches:
            return 0, matched_tokens, reasons
        if apply_neg_scope:
            scores, negated = self.apply_negation_scope(tokens, matches)
            if negated:
                reasons.append("부정어 스코프")
        else:
            scores = [self.senti_dict.get(match["token"], 0) for match in matches]
        raw_score = sum(scores)
        return raw_score, matched_tokens, reasons

    def bin_sentiment_score(self, raw_score: float) -> int:
        if raw_score <= -2:
            return -2
        if raw_score == -1:
            return -1
        if raw_score == 0:
            return 0
        if raw_score == 1:
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
        if not self.brand_map or not isinstance(text, str):
            return "전체"
        lowered = text.lower()
        for topic, keywords in self.brand_map.items():
            for keyword in keywords:
                if keyword.lower() in lowered:
                    return topic
        return "전체"

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
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in df["full_text"]
        ))
        series = pd.Series(tokens)
        freq = series.value_counts()
        self.tbl_token_sample.setRowCount(len(freq))
        for row_idx, (token, count) in enumerate(freq.items()):
            self.tbl_token_sample.setItem(row_idx, 0, QTableWidgetItem(token))
            self.tbl_token_sample.setItem(row_idx, 1, QTableWidgetItem(str(count)))

    def tokenize_text(self, text: str):
        if not isinstance(text, str):
            return []
        analysis = self.kiwi.analyze(text)
        if not analysis or not analysis[0][0]:
            return []
        tokens = [token for token, _, _, _ in analysis[0][0]]
        cleaned = []
        for token in tokens:
            if token in self.stopwords:
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
            cleaned.append(token)
        return cleaned

    def parse_custom_terms(self, raw_text: str):
        if not raw_text:
            return set()
        tokens = [token.strip() for token in re.split(r"[,\n]", raw_text) if token.strip()]
        return set(tokens)

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
        shapes = ["circle", "diamond", "triangle", "square"]
        colormap = random.choice(colormaps)
        shape = random.choice(shapes)
        mask = self.build_wordcloud_mask(shape)
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
        else:
            df["bucket"] = df["date"].dt.to_period("Y").dt.start_time

        selected_page = self.cb_page_type_filter.currentText()
        if selected_page != "전체":
            df = df[df["page_type"] == selected_page]

        if self.chk_split_by_page_type.isChecked():
            summary = df.groupby(["bucket", "page_type"]).size().reset_index(name="count")
        else:
            summary = df.groupby("bucket").size().reset_index(name="count")
            summary["page_type"] = "전체"

        summary = summary.sort_values("bucket")
        self.buzz_df = summary

        metric = self.cb_buzz_metric.currentText()
        labels = [self.format_date(val) for val in summary["bucket"]]
        if self.chk_split_by_page_type.isChecked():
            pivot = summary.pivot_table(
                index="bucket", columns="page_type", values="count", fill_value=0
            )
            if metric == "%":
                pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
                ylabel = "%"
            else:
                ylabel = "count"
            labels = [self.format_date(val) for val in pivot.index]
            series = [(str(col), pivot[col].tolist()) for col in pivot.columns]
            self.buzz_canvas.plot_stacked_bar(labels, series, "버즈량", ylabel)
        else:
            values = summary["count"].tolist()
            if metric == "%":
                total = sum(values) or 1
                values = [round(val / total * 100, 2) for val in values]
                ylabel = "%"
            else:
                ylabel = "count"
            self.buzz_canvas.plot_bar(labels, values, "버즈량", ylabel)
        self.chart_images["buzz"] = self.save_chart(self.buzz_canvas, "buzz")

        self.tbl_buzz.setRowCount(len(summary))
        for row_idx, (_, row) in enumerate(summary.iterrows()):
            self.tbl_buzz.setItem(row_idx, 0, QTableWidgetItem(self.format_date(row["bucket"])))
            self.tbl_buzz.setItem(row_idx, 1, QTableWidgetItem(str(row["page_type"])))
            self.tbl_buzz.setItem(row_idx, 2, QTableWidgetItem(str(row["count"])))

    def build_wordcloud(self):
        if self.df_clean is None:
            return
        topn = int(self.cb_wc_topn.currentText())
        df = self.filter_df_by_period(
            self.df_clean, self.cb_wc_period_unit, self.cb_wc_period_value
        )
        df = self.apply_monthly_sampling(df)
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in df["full_text"]
        ))
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
            colormap, mask = "Blues", None
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            font_path=str(font_path) if font_path else None,
            colormap=colormap,
            mask=mask,
        )
        wc_img = wordcloud.generate_from_frequencies(freq.head(topn).to_dict())
        wc_path = os.path.join(os.getcwd(), "wordcloud.png")
        wc_img.to_file(wc_path)
        self.wc_image_path = wc_path
        self.chart_images["wordcloud"] = wc_path

        pixmap = QPixmap(wc_path)
        self.lbl_wc_view.setPixmap(pixmap.scaled(self.lbl_wc_view.size(), Qt.KeepAspectRatio))

        top_freq = freq.head(topn)
        self.tbl_wc_topn.setRowCount(len(top_freq))
        for row_idx, (token, count) in enumerate(top_freq.items()):
            self.tbl_wc_topn.setItem(row_idx, 0, QTableWidgetItem(token))
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
        if scope == "문서(로우)":
            for text in df["full_text"]:
                tokens = [normalize_term(token) for token in self.tokenize_text(text)]
                tokens = [token for token in tokens if token and token not in self.network_stopwords]
                if tokens:
                    token_lists.append(tokens)
        else:
            for text in df["full_text"]:
                for sentence in split_sentences(text):
                    tokens = [normalize_term(token) for token in self.tokenize_text(sentence)]
                    tokens = [token for token in tokens if token and token not in self.network_stopwords]
                    if tokens:
                        token_lists.append(tokens)

        if not token_lists:
            self.statusBar().showMessage("네트워크 생성에 필요한 토큰이 없습니다.")
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
        if self.network_level_map:
            level_palette = {
                1: "#ff6fae",
                2: "#4a4a4a",
                3: "#9e9e9e",
            }
            node_colors = [
                level_palette.get(self.network_level_map.get(node, 3), "#9e9e9e")
                for node in graph.nodes
            ]
        else:
            norm = colors.Normalize(vmin=0, vmax=max_degree or 1)
            cmap = cm.get_cmap("Set2")
            node_colors = [cmap(norm(deg)) for deg in degrees]
        edge_weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [0.6 + 3.0 * (weight / max_weight) for weight in edge_weights]
        nx.draw_networkx_nodes(
            graph, pos, ax=self.network_canvas.ax, node_size=sizes, node_color=node_colors
        )
        nx.draw_networkx_edges(
            graph, pos, ax=self.network_canvas.ax, width=edge_widths, edge_color="#999999", alpha=0.7
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
        self.update_sentiment_lexicon()

        df = self.apply_monthly_sampling(self.df_clean)
        records = []
        for _, row in df.iterrows():
            text = row.get("full_text", "")
            prev_final_score = None
            for sentence in self.split_sentiment_sentences(text):
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
                    tokens = self.tokenize_text(clause)
                    matches = self.match_sentiment_tokens(tokens)
                    clause_score, clause_matched, clause_reasons = self.calculate_sentence_score(
                        clause, tokens, matches, use_neg_scope
                    )
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
        mode = self.cb_sent_mode.currentText()
        topic_filter = self.cb_brand_filter.currentText()
        self.cb_brand_filter.setEnabled(mode == "사전별 감성")
        if mode == "사전별 감성" and topic_filter != "전체":
            df = df[df["topic"] == topic_filter]

        view = self.cb_sent_view.currentText()
        if view == "월별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m")
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
        for row_idx, (_, row) in enumerate(show_records.iterrows()):
            self.tbl_sent_records.setItem(row_idx, 0, QTableWidgetItem(self.format_date(row["date"])))
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
        if "topic" in summary.columns:
            pivot = summary.pivot_table(
                index="score", columns="topic", values="count", fill_value=0
            ).reindex(score_order, fill_value=0)
            if metric == "%":
                pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
                ylabel = "%"
            else:
                ylabel = "count"
            series = [(str(col), pivot[col].tolist()) for col in pivot.columns]
            self.sent_canvas.plot_stacked_bar(labels, series, "감성 분포", ylabel)
        else:
            counts = {score: 0 for score in score_order}
            for _, row in summary.iterrows():
                counts[row["score"]] = int(row["count"])
            values = [counts[score] for score in score_order]
            if metric == "%":
                total = sum(values) or 1
                values = [round(value / total * 100, 2) for value in values]
                ylabel = "%"
            else:
                ylabel = "count"
            self.sent_canvas.ax.clear()
            bar_colors = [SENTIMENT_COLORS[score] for score in score_order]
            if labels:
                self.sent_canvas.ax.bar(labels, values, color=bar_colors)
                self.sent_canvas.ax.set_xticks(range(len(labels)))
                self.sent_canvas.ax.set_xticklabels(labels, rotation=45, ha="right")
            self.sent_canvas.ax.set_title("감성 분포")
            self.sent_canvas.ax.set_ylabel(ylabel)
            self.sent_canvas.ax.grid(axis="y", alpha=0.3)
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
        topics = sorted(df["topic"].dropna().unique())
        if not topics:
            return
        for topic in topics:
            sub = df[df["topic"] == topic]
            counts = (
                sub.groupby("score")
                .size()
                .reindex(score_order, fill_value=0)
                .astype(float)
            )
            total = counts.sum() or 1.0
            percents = [round(value / total * 100, 2) for value in counts.tolist()]
            canvas = ChartCanvas()
            canvas.ax.clear()
            x_positions = [0]
            bottoms = [0.0]
            for score, label, value in zip(score_order, labels, percents):
                values = [value]
                color = SENTIMENT_COLORS.get(score, "#999999")
                bars = canvas.ax.bar(x_positions, values, bottom=bottoms, label=label, color=color)
                for bar, bottom, value in zip(bars, bottoms, values):
                    if value <= 0:
                        continue
                    canvas.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value >= 15 else "#333333",
                    )
                bottoms = [b + v for b, v in zip(bottoms, values)]
            canvas.ax.set_xticks(x_positions)
            canvas.ax.set_xticklabels([topic])
            canvas.ax.set_ylim(0, 100)
            canvas.ax.set_title(f"{topic} 감성 비율")
            canvas.ax.set_ylabel("%")
            canvas.ax.legend(loc="upper right")
            canvas.figure.tight_layout()
            canvas.draw()
            self.sent_topic_charts_layout.addWidget(canvas)

    def plot_sentiment_trend(self, df):
        if df is None or df.empty:
            self.sent_trend_canvas.ax.clear()
            self.sent_trend_canvas.draw()
            return
        view = self.cb_sent_view.currentText()
        if view == "월별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m")
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
            if "network_nodes" in items and self.nodes_view_df is not None:
                self.nodes_view_df.to_excel(writer, sheet_name="network_nodes", index=False)
            if "network_edges" in items and self.edges_view_df is not None:
                self.edges_view_df.to_excel(writer, sheet_name="network_edges", index=False)
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
