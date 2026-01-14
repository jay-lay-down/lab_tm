import re
import sys
from dataclasses import dataclass
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import requests
from kiwipiepy import Kiwi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Qt5Agg")

KNU_DICT_URL = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json"


@dataclass
class AnalysisResult:
    records: pd.DataFrame
    summary: pd.DataFrame


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


class ChartCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(6, 4))
        super().__init__(fig)
        self.setParent(parent)

    def plot_bar(self, labels, values, title, ylabel):
        self.ax.clear()
        if labels:
            self.ax.bar(labels, values, color="#4b77be")
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
        self.resize(1200, 800)

        self.df = None
        self.analysis_result = None
        self.senti_dict = None
        self.kiwi = Kiwi()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._build_dictionary_box())
        layout.addWidget(self._build_controls())
        layout.addWidget(self._build_view_options())
        layout.addWidget(self._build_split_view())

    def _build_dictionary_box(self):
        box = QGroupBox("브랜드/컨셉 사전 설정")
        layout = QVBoxLayout(box)
        self.brand_text = QTextEdit()
        self.brand_text.setPlaceholderText(
            "브랜드: 키워드1|키워드2\n예) 삼성: 삼성|갤럭시\n샤오미: 샤오미|미지아"
        )
        layout.addWidget(self.brand_text)
        return box

    def _build_controls(self):
        box = QGroupBox("데이터 로드")
        layout = QGridLayout(box)

        self.file_label = QLabel("파일 미선택")
        load_btn = QPushButton("엑셀 파일 열기")
        load_btn.clicked.connect(self.load_file)

        self.text_col_input = QLineEdit("content")
        self.date_col_input = QLineEdit("Date")
        self.channel_col_input = QLineEdit("page_type")

        analyze_btn = QPushButton("분석 실행")
        analyze_btn.clicked.connect(self.run_analysis)

        layout.addWidget(load_btn, 0, 0)
        layout.addWidget(self.file_label, 0, 1, 1, 3)
        layout.addWidget(QLabel("텍스트 컬럼"), 1, 0)
        layout.addWidget(self.text_col_input, 1, 1)
        layout.addWidget(QLabel("날짜 컬럼"), 1, 2)
        layout.addWidget(self.date_col_input, 1, 3)
        layout.addWidget(QLabel("채널 컬럼"), 2, 0)
        layout.addWidget(self.channel_col_input, 2, 1)
        layout.addWidget(analyze_btn, 2, 3)

        return box

    def _build_view_options(self):
        box = QGroupBox("보기 옵션")
        layout = QHBoxLayout(box)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["전체", "월별", "채널별"])
        self.view_combo.currentIndexChanged.connect(self.refresh_views)

        self.brand_combo = QComboBox()
        self.brand_combo.addItem("전체")
        self.brand_combo.currentIndexChanged.connect(self.refresh_views)

        layout.addWidget(QLabel("보기"))
        layout.addWidget(self.view_combo)
        layout.addWidget(QLabel("브랜드"))
        layout.addWidget(self.brand_combo)
        layout.addStretch()

        return box

    def _build_split_view(self):
        splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "브랜드",
            "감성 점수",
            "문장",
            "날짜",
            "채널",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.chart = ChartCanvas()

        splitter.addWidget(self.table)
        splitter.addWidget(self.chart)
        splitter.setSizes([700, 500])

        return splitter

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "엑셀 파일 선택", "", "Excel Files (*.xlsx *.xls)"
        )
        if not file_path:
            return
        self.df = pd.read_excel(file_path)
        self.file_label.setText(file_path)

    def run_analysis(self):
        if self.df is None:
            self.statusBar().showMessage("데이터를 먼저 불러주세요.")
            return

        brand_map = parse_brand_dictionary(self.brand_text.toPlainText())
        if not brand_map:
            self.statusBar().showMessage("브랜드/컨셉 사전을 입력해주세요.")
            return

        text_col = self.text_col_input.text().strip()
        date_col = self.date_col_input.text().strip()
        channel_col = self.channel_col_input.text().strip()

        missing_cols = [col for col in [text_col, date_col] if col and col not in self.df.columns]
        if missing_cols:
            self.statusBar().showMessage(f"컬럼이 없습니다: {', '.join(missing_cols)}")
            return

        if self.senti_dict is None:
            try:
                self.senti_dict = load_knu_dictionary()
            except Exception as exc:
                self.statusBar().showMessage(f"사전 로드 실패: {exc}")
                return

        self.analysis_result = self.analyze_data(
            df=self.df,
            brand_map=brand_map,
            text_col=text_col,
            date_col=date_col,
            channel_col=channel_col,
        )
        self.update_brand_options(sorted(brand_map.keys()))
        self.refresh_views()
        self.statusBar().showMessage("분석 완료")

    def analyze_data(self, df, brand_map, text_col, date_col, channel_col):
        records = []
        for _, row in df.iterrows():
            text = row.get(text_col, "")
            date_val = row.get(date_col)
            channel_val = row.get(channel_col) if channel_col in df.columns else ""

            try:
                date_obj = pd.to_datetime(date_val, errors="coerce")
            except Exception:
                date_obj = pd.NaT

            for sentence in split_sentences(text):
                sentence_lower = sentence.lower()
                matched_brands = []
                for brand, keywords in brand_map.items():
                    for keyword in keywords:
                        if keyword.lower() in sentence_lower:
                            matched_brands.append(brand)
                            break

                if not matched_brands:
                    continue

                tokens = self.kiwi.analyze(sentence)
                token_scores = []
                for token, pos, _, _ in tokens[0][0]:
                    token_scores.append(self.senti_dict.get(token, 0))
                score = sum(token_scores)

                for brand in matched_brands:
                    records.append(
                        {
                            "brand": brand,
                            "score": score,
                            "sentence": sentence,
                            "date": date_obj,
                            "channel": channel_val,
                            "month": date_obj.strftime("%Y-%m") if pd.notnull(date_obj) else "",
                        }
                    )

        records_df = pd.DataFrame(records)
        summary_df = self.build_summary(records_df)
        return AnalysisResult(records=records_df, summary=summary_df)

    def build_summary(self, records_df):
        if records_df.empty:
            return pd.DataFrame()
        summary = (
            records_df.groupby("brand")["score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_score", "count": "mentions"})
        )
        return summary

    def update_brand_options(self, brands):
        self.brand_combo.blockSignals(True)
        self.brand_combo.clear()
        self.brand_combo.addItem("전체")
        for brand in brands:
            self.brand_combo.addItem(brand)
        self.brand_combo.blockSignals(False)

    def refresh_views(self):
        if not self.analysis_result:
            return
        view_mode = self.view_combo.currentText()
        brand_filter = self.brand_combo.currentText()
        records = self.analysis_result.records

        if records.empty:
            self.table.setRowCount(0)
            self.chart.plot_bar([], [], "", "")
            return

        if brand_filter != "전체":
            records = records[records["brand"] == brand_filter]

        if view_mode == "전체":
            summary = (
                records.groupby("brand")["score"]
                .mean()
                .reset_index()
                .sort_values("score", ascending=False)
            )
            labels = summary["brand"].tolist()
            values = summary["score"].tolist()
            self.chart.plot_bar(labels, values, "브랜드별 평균 감성 점수", "평균 점수")
        elif view_mode == "월별":
            if records["month"].eq("").all():
                self.chart.plot_bar([], [], "날짜 정보가 없습니다", "")
            else:
                summary = (
                    records.groupby("month")["score"]
                    .mean()
                    .reset_index()
                    .sort_values("month")
                )
                labels = summary["month"].tolist()
                values = summary["score"].tolist()
                self.chart.plot_bar(labels, values, "월별 평균 감성 점수", "평균 점수")
        else:
            if "channel" not in records.columns or records["channel"].eq("").all():
                self.chart.plot_bar([], [], "채널 정보가 없습니다", "")
            else:
                summary = (
                    records.groupby("channel")["score"]
                    .mean()
                    .reset_index()
                    .sort_values("score", ascending=False)
                )
                labels = summary["channel"].tolist()
                values = summary["score"].tolist()
                self.chart.plot_bar(labels, values, "채널별 평균 감성 점수", "평균 점수")

        self.populate_table(records)

    def populate_table(self, records):
        show_records = records.copy()
        show_records = show_records.sort_values("score", ascending=False).head(200)
        self.table.setRowCount(len(show_records))

        for row_idx, (_, row) in enumerate(show_records.iterrows()):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row.get("brand", ""))))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(row.get("score", ""))))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(row.get("sentence", ""))))
            date_val = row.get("date")
            date_text = ""
            if isinstance(date_val, (datetime, pd.Timestamp)) and pd.notnull(date_val):
                date_text = date_val.strftime("%Y-%m-%d")
            self.table.setItem(row_idx, 3, QTableWidgetItem(date_text))
            self.table.setItem(row_idx, 4, QTableWidgetItem(str(row.get("channel", ""))))


def main():
    app = QApplication(sys.argv)
    window = TextMiningApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
