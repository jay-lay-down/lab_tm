import itertools
import os
import re
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
from kiwipiepy import Kiwi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from openpyxl.drawing.image import Image as XLImage
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
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
    QPushButton,
    QSplitter,
    QSpinBox,
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


def load_knu_dictionary():
    response = requests.get(KNU_DICT_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    senti_dict = {}
    for entry in data:
        root = entry.get("word_root", entry.get("word"))
        score = int(entry.get("polarity", 0))
        senti_dict[root] = score
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


def normalize_column_name(name: str):
    return re.sub(r"[\s_]", "", name.lower())


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


class TextMiningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Mining Tool - PyQt")
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
        self.nodes_df = None
        self.edges_df = None
        self.nodes_view_df = None
        self.edges_view_df = None
        self.sentiment_records_df = None
        self.sentiment_summary_df = None
        self.chart_images = {}

        self.senti_dict = None
        self.kiwi = Kiwi()

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
        self.cb_granularity.addItems(["일", "주", "월", "분기", "연도"])
        self.chk_split_by_page_type = QCheckBox("page_type 분리")
        self.sb_topn_page_type = QSpinBox()
        self.sb_topn_page_type.setRange(1, 20)
        self.sb_topn_page_type.setValue(5)
        self.btn_refresh_buzz = QPushButton("버즈 계산")
        self.btn_refresh_buzz.clicked.connect(self.build_buzz)

        top_layout.addWidget(self.cb_granularity, 0, 0)
        top_layout.addWidget(self.chk_split_by_page_type, 0, 1)
        top_layout.addWidget(self.sb_topn_page_type, 0, 2)
        top_layout.addWidget(self.btn_refresh_buzz, 0, 4)

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
        splitter.setSizes([600, 600])
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.txt_brand_dict = QTextEdit()
        self.txt_brand_dict.setPlaceholderText("브랜드: 키워드1|키워드2")
        self.btn_apply_brand = QPushButton("브랜드 사전 적용")
        self.btn_apply_brand.clicked.connect(self.apply_brand_dict)
        left_layout.addWidget(self.txt_brand_dict)
        left_layout.addWidget(self.btn_apply_brand)

        right_split = QSplitter(Qt.Vertical)
        right_split.setSizes([420, 220])

        top_right = QWidget()
        top_layout = QVBoxLayout(top_right)
        self.txt_stopwords = QTextEdit()
        self.txt_stopwords.setPlaceholderText("불용어를 줄바꿈으로 입력")
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
        self.btn_apply_stopwords = QPushButton("불용어/옵션 적용")
        self.btn_apply_stopwords.clicked.connect(self.apply_stopwords)

        top_layout.addWidget(self.txt_stopwords)
        top_layout.addLayout(opts_row)
        top_layout.addWidget(self.btn_apply_stopwords)

        self.tbl_token_sample = QTableWidget()
        self.tbl_token_sample.setColumnCount(2)
        self.tbl_token_sample.setHorizontalHeaderLabels(["token", "count"])
        self.tbl_token_sample.horizontalHeader().setStretchLastSection(True)

        right_split.addWidget(top_right)
        right_split.addWidget(self.tbl_token_sample)

        splitter.addWidget(left)
        splitter.addWidget(right_split)

        self.tabs.addTab(tab, "텍스트마이닝 설정")

    def _build_tab_wordcloud(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QWidget()
        top.setFixedHeight(70)
        top_layout = QGridLayout(top)
        self.cb_wc_topn = QComboBox()
        self.cb_wc_topn.addItems(["30", "50", "100", "200"])
        self.btn_build_wc = QPushButton("워드클라우드 생성")
        self.btn_build_wc.clicked.connect(self.build_wordcloud)
        top_layout.addWidget(self.cb_wc_topn, 0, 0)
        top_layout.addWidget(self.btn_build_wc, 0, 1)

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
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["기본", "고급"])
        self.cb_mode.currentIndexChanged.connect(self.toggle_network_advanced)
        self.sb_min_node_count = QSpinBox()
        self.sb_min_node_count.setRange(1, 100)
        self.sb_min_node_count.setValue(5)
        self.sb_min_edge_weight = QSpinBox()
        self.sb_min_edge_weight.setRange(1, 100)
        self.sb_min_edge_weight.setValue(3)
        self.sb_max_nodes = QSpinBox()
        self.sb_max_nodes.setRange(50, 1000)
        self.sb_max_nodes.setValue(300)

        self.le_node_search = QLineEdit()
        self.le_node_search.setPlaceholderText("노드 검색")
        self.btn_add_seed = QPushButton("Seed 추가")
        self.btn_add_seed.clicked.connect(self.add_seed)
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
        self.lbl_advanced_hint = QLabel(
            "PMI는 희귀 단어쌍을 과대평가할 수 있어 min_node_count ≥ 10, "
            "min_edge_weight ≥ 5를 권장합니다. (데이터가 작을수록 필터를 높이세요)"
        )

        top_layout.addWidget(self.btn_build_graph, 0, 0)
        top_layout.addWidget(self.cb_mode, 0, 1)
        top_layout.addWidget(self.sb_min_node_count, 0, 2)
        top_layout.addWidget(self.sb_min_edge_weight, 0, 3)
        top_layout.addWidget(self.sb_max_nodes, 0, 4)
        top_layout.addWidget(self.le_node_search, 1, 0)
        top_layout.addWidget(self.btn_add_seed, 1, 1)
        top_layout.addWidget(self.cb_hop_depth, 1, 2)
        top_layout.addWidget(self.btn_apply_hop, 1, 3)
        top_layout.addWidget(self.btn_reset_view, 1, 4)
        top_layout.addWidget(self.cb_cooc_scope, 2, 0, 1, 2)
        top_layout.addWidget(self.cb_weight_mode, 2, 2, 1, 2)
        top_layout.addWidget(self.lbl_advanced_hint, 2, 4)

        splitter.addWidget(top)

        mid = QSplitter(Qt.Horizontal)
        mid.setSizes([650, 350])
        self.list_nodes_ranked = QListWidget()
        self.list_nodes_ranked.itemDoubleClicked.connect(self.add_seed_from_list)
        self.list_seed_nodes = QListWidget()
        mid.addWidget(self.list_nodes_ranked)
        mid.addWidget(self.list_seed_nodes)
        splitter.addWidget(mid)

        bottom = QSplitter(Qt.Horizontal)
        bottom.setSizes([700, 500])
        self.network_canvas = ChartCanvas()
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
        top.setFixedHeight(80)
        top_layout = QGridLayout(top)
        self.cb_sent_view = QComboBox()
        self.cb_sent_view.addItems(["전체", "월별", "채널별"])
        self.cb_sent_view.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_sent_metric = QComboBox()
        self.cb_sent_metric.addItems(["count", "%"])
        self.cb_sent_metric.currentIndexChanged.connect(self.update_sentiment_view)
        self.cb_brand_filter = QComboBox()
        self.cb_brand_filter.addItem("전체")
        self.cb_brand_filter.currentIndexChanged.connect(self.update_sentiment_view)
        self.btn_run_sentiment = QPushButton("감성분석 실행")
        self.btn_run_sentiment.clicked.connect(self.run_sentiment)

        top_layout.addWidget(self.cb_sent_view, 0, 0)
        top_layout.addWidget(self.cb_sent_metric, 0, 1)
        top_layout.addWidget(self.cb_brand_filter, 0, 2)
        top_layout.addWidget(self.btn_run_sentiment, 0, 4)

        layout.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([720, 480])
        self.tbl_sent_records = QTableWidget()
        self.tbl_sent_records.setColumnCount(5)
        self.tbl_sent_records.setHorizontalHeaderLabels([
            "date",
            "page_type",
            "text",
            "score",
            "label",
        ])
        self.tbl_sent_records.horizontalHeader().setStretchLastSection(True)
        self.sent_canvas = ChartCanvas()
        splitter.addWidget(self.tbl_sent_records)
        splitter.addWidget(self.sent_canvas)
        layout.addWidget(splitter)

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
        if isinstance(value, (datetime, pd.Timestamp)) and pd.notnull(value):
            return value.strftime("%Y-%m-%d")
        return ""

    def apply_brand_dict(self):
        self.brand_map = parse_brand_dictionary(self.txt_brand_dict.toPlainText())
        self.cb_brand_filter.blockSignals(True)
        self.cb_brand_filter.clear()
        self.cb_brand_filter.addItem("전체")
        for brand in sorted(self.brand_map.keys()):
            self.cb_brand_filter.addItem(brand)
        self.cb_brand_filter.blockSignals(False)
        self.statusBar().showMessage("브랜드 사전을 적용했습니다.")

    def apply_stopwords(self):
        self.stopwords = {line.strip() for line in self.txt_stopwords.toPlainText().splitlines() if line.strip()}
        self.clean_opts = {
            "remove_numbers": self.chk_remove_numbers.isChecked(),
            "remove_symbols": self.chk_remove_symbols.isChecked(),
            "remove_single": self.chk_remove_single.isChecked(),
            "korean_only": self.chk_korean_only.isChecked(),
            "english_only": self.chk_english_only.isChecked(),
        }
        self.statusBar().showMessage("불용어/옵션을 적용했습니다.")
        self.refresh_token_sample()

    def refresh_token_sample(self):
        if self.df_clean is None:
            return
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in self.df_clean["full_text"].head(100)
        ))
        series = pd.Series(tokens)
        freq = series.value_counts().head(50)
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

    def build_buzz(self):
        if self.df_clean is None:
            return
        df = self.df_clean.copy()
        gran = self.cb_granularity.currentText()
        if gran == "일":
            df["bucket"] = df["date"]
        elif gran == "주":
            df["bucket"] = df["date"].dt.to_period("W").dt.start_time
        elif gran == "월":
            df["bucket"] = df["date"].dt.to_period("M").dt.start_time
        elif gran == "분기":
            df["bucket"] = df["date"].dt.to_period("Q").dt.start_time
        else:
            df["bucket"] = df["date"].dt.to_period("Y").dt.start_time

        if self.chk_split_by_page_type.isChecked():
            topn = self.sb_topn_page_type.value()
            top_pages = df["page_type"].value_counts().head(topn).index
            df = df[df["page_type"].isin(top_pages)]
            summary = df.groupby(["bucket", "page_type"]).size().reset_index(name="count")
        else:
            summary = df.groupby("bucket").size().reset_index(name="count")
            summary["page_type"] = "전체"

        summary = summary.sort_values("bucket")
        self.buzz_df = summary

        labels = [self.format_date(val) for val in summary["bucket"]]
        values = summary["count"].tolist()
        self.buzz_canvas.plot_line(labels, values, "버즈량", "count")
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
        tokens = list(itertools.chain.from_iterable(
            self.tokenize_text(text) for text in self.df_clean["full_text"]
        ))
        series = pd.Series(tokens)
        freq = series.value_counts().head(topn)
        self.word_freq_df = freq.reset_index().rename(columns={"index": "token", 0: "count"})

        if freq.empty:
            self.lbl_wc_view.setText("데이터가 없습니다")
            return
        wordcloud = WordCloud(width=800, height=500, background_color="white")
        wc_img = wordcloud.generate_from_frequencies(freq.to_dict())
        wc_path = os.path.join(os.getcwd(), "wordcloud.png")
        wc_img.to_file(wc_path)
        self.wc_image_path = wc_path
        self.chart_images["wordcloud"] = wc_path

        pixmap = QPixmap(wc_path)
        self.lbl_wc_view.setPixmap(pixmap.scaled(self.lbl_wc_view.size(), Qt.KeepAspectRatio))

        self.tbl_wc_topn.setRowCount(len(freq))
        for row_idx, (token, count) in enumerate(freq.items()):
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
        if self.cb_mode.currentText() == "기본":
            scope = "문서(로우)"
            weight_mode = "count"
        else:
            scope = self.cb_cooc_scope.currentText()
            weight_mode = self.cb_weight_mode.currentText()

        token_lists = []
        if scope == "문서(로우)":
            for text in self.df_clean["full_text"]:
                tokens = self.tokenize_text(text)
                if len(tokens) > 1:
                    token_lists.append(tokens)
        else:
            for text in self.df_clean["full_text"]:
                for sentence in split_sentences(text):
                    tokens = self.tokenize_text(sentence)
                    if len(tokens) > 1:
                        token_lists.append(tokens)

        if not token_lists:
            self.statusBar().showMessage("네트워크 생성에 필요한 토큰이 없습니다.")
            return

        node_counts = {}
        edge_counts = {}
        for tokens in token_lists:
            unique_tokens = list(dict.fromkeys(tokens))
            for token in unique_tokens:
                node_counts[token] = node_counts.get(token, 0) + 1
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

        min_node = self.sb_min_node_count.value()
        min_edge = self.sb_min_edge_weight.value()
        max_nodes = self.sb_max_nodes.value()
        filtered_nodes = {node for node, count in node_counts.items() if count >= min_node}
        filtered_edges = {
            edge: weight
            for edge, weight in edge_weights.items()
            if edge[0] in filtered_nodes and edge[1] in filtered_nodes and weight >= min_edge
        }

        ranked_nodes = sorted(filtered_nodes, key=lambda n: node_counts[n], reverse=True)[:max_nodes]
        ranked_nodes_set = set(ranked_nodes)
        filtered_edges = {
            edge: weight
            for edge, weight in filtered_edges.items()
            if edge[0] in ranked_nodes_set and edge[1] in ranked_nodes_set
        }

        graph = nx.Graph()
        for node in ranked_nodes:
            graph.add_node(node, count=node_counts[node])
        for (a, b), weight in filtered_edges.items():
            graph.add_edge(a, b, weight=weight)

        self.graph_full = graph
        self.graph_view = graph.copy()
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

    def add_seed_from_list(self, item):
        self.list_seed_nodes.addItem(item.text())

    def add_seed(self):
        text = self.le_node_search.text().strip()
        if text:
            self.list_seed_nodes.addItem(text)

    def apply_hop(self):
        if self.graph_full is None:
            return
        seeds = [self.list_seed_nodes.item(i).text() for i in range(self.list_seed_nodes.count())]
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

    def reset_network_view(self):
        if self.graph_full is None:
            return
        self.graph_view = self.graph_full.copy()
        self.nodes_view_df = self.nodes_df.copy()
        self.edges_view_df = self.edges_df.copy()
        self.draw_network(self.graph_view)
        self.populate_network_tables(self.edges_view_df, self.nodes_view_df)
        self.chart_images["network"] = self.save_chart(self.network_canvas, "network")

    def draw_network(self, graph):
        self.network_canvas.ax.clear()
        if graph is None or graph.number_of_nodes() == 0:
            self.network_canvas.draw()
            return
        pos = nx.spring_layout(graph, k=0.6, seed=42)
        nx.draw_networkx_nodes(graph, pos, ax=self.network_canvas.ax, node_size=200, node_color="#6baed6")
        nx.draw_networkx_edges(graph, pos, ax=self.network_canvas.ax, width=1.0, edge_color="#999999")
        nx.draw_networkx_labels(graph, pos, ax=self.network_canvas.ax, font_size=8)
        self.network_canvas.ax.set_title("네트워크")
        self.network_canvas.ax.axis("off")
        self.network_canvas.draw()

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
        if self.senti_dict is None:
            self.senti_dict = load_knu_dictionary()

        records = []
        for _, row in self.df_clean.iterrows():
            text = row.get("full_text", "")
            tokens = self.tokenize_text(text)
            score = sum(self.senti_dict.get(token, 0) for token in tokens)
            label = "중립"
            if score > 0:
                label = "긍정"
            elif score < 0:
                label = "부정"
            brand = self.match_brand(text)
            records.append(
                {
                    "date": row.get("date"),
                    "page_type": row.get("page_type"),
                    "text": text,
                    "score": score,
                    "label": label,
                    "brand": brand,
                }
            )

        self.sentiment_records_df = pd.DataFrame(records)
        self.update_sentiment_view()

    def update_sentiment_view(self):
        if self.sentiment_records_df is None or self.sentiment_records_df.empty:
            return
        df = self.sentiment_records_df.copy()
        brand_filter = self.cb_brand_filter.currentText()
        if brand_filter != "전체":
            df = df[df["brand"] == brand_filter]

        view = self.cb_sent_view.currentText()
        if view == "월별":
            df["bucket"] = df["date"].dt.strftime("%Y-%m")
        elif view == "채널별":
            df["bucket"] = df["page_type"].fillna("")
        else:
            df["bucket"] = "전체"

        summary = df.groupby(["bucket", "label"]).size().reset_index(name="count")
        self.sentiment_summary_df = summary
        self.populate_sentiment_table(df)
        self.plot_sentiment_chart(summary)

    def populate_sentiment_table(self, df):
        show_records = df.head(200)
        self.tbl_sent_records.setRowCount(len(show_records))
        for row_idx, (_, row) in enumerate(show_records.iterrows()):
            self.tbl_sent_records.setItem(row_idx, 0, QTableWidgetItem(self.format_date(row["date"])))
            self.tbl_sent_records.setItem(row_idx, 1, QTableWidgetItem(str(row["page_type"])))
            self.tbl_sent_records.setItem(row_idx, 2, QTableWidgetItem(str(row["text"])))
            self.tbl_sent_records.setItem(row_idx, 3, QTableWidgetItem(str(row["score"])))
            self.tbl_sent_records.setItem(row_idx, 4, QTableWidgetItem(str(row["label"])))

    def plot_sentiment_chart(self, summary):
        metric = self.cb_sent_metric.currentText()
        if metric == "%":
            total = summary["count"].sum()
            summary["value"] = (summary["count"] / total * 100).round(2)
            ylabel = "%"
        else:
            summary["value"] = summary["count"]
            ylabel = "count"
        labels = [f"{bucket}-{label}" for bucket, label in summary[["bucket", "label"]].values]
        values = summary["value"].tolist()
        self.sent_canvas.plot_bar(labels, values, "감성분석", ylabel)
        self.chart_images["sentiment"] = self.save_chart(self.sent_canvas, "sentiment")

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

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
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
    window = TextMiningApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
