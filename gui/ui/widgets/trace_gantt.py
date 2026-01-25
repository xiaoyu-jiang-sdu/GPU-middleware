import json
from collections import defaultdict
from pathlib import Path

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TraceGanttWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("推理时间线")   # ← 标题

        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def load_trace(self, trace_path: Path, model="", backend=""):
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)

        with trace_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        self._render(events, model, backend)

    def _render(self, events, model="", backend=""):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 仅区间事件
        events = [e for e in events if e.get("ph") == "X"]
        if not events:
            self.canvas.draw_idle()
            return

        # 过滤极短事件
        events = [e for e in events if e.get("dur", 0) >= 50.0]

        lanes = defaultdict(list)
        for e in events:
            lane = e.get("args", {}).get("op", e["name"])
            lanes[lane].append(e)

        t0 = min(e["ts"] for e in events)

        yticks, ylabels = [], []
        y = 0
        height = 0.6

        for lane_name, evs in list(lanes.items())[:30]:
            for e in evs:
                start = (e["ts"] - t0) / 1000.0
                dur = e["dur"] / 1000.0
                ax.barh(
                    y,
                    dur,
                    left=start,
                    height=height,
                    color=self._color_for(lane_name),
                    alpha=0.8
                )

            yticks.append(y)
            ylabels.append(lane_name)
            y += 1

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Time (ms)")

        title = "Inference Timeline"
        if model:
            title = f"{model} on {backend}"
        ax.set_title(title)

        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        # 给 y 轴标签留空间
        self.figure.subplots_adjust(left=0.25)

        self.canvas.draw_idle()

    @staticmethod
    def _color_for(name: str):
        palette = [
            "#60A5FA", "#34D399", "#FBBF24",
            "#F87171", "#A78BFA", "#FB7185",
            "#2DD4BF", "#F97316"
        ]
        return palette[hash(name) % len(palette)]
