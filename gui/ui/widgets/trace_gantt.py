import json
from collections import defaultdict
from pathlib import Path

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TraceGanttWidget(QGroupBox):
    def __init__(self):
        super().__init__("推理时间线") 

        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self._bars = []
        self._annot = None
        self._lane_durations = defaultdict(float)  # lane 总时间
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def load_trace(self, trace_path: Path, model="", backend="", batch_size=""):
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)

        with trace_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("traceEvents", [])
        self._render(events, model, backend, batch_size)

    def _render(self, events, model="", backend="", batch_size=""):
        self._bars.clear()
        self._lane_durations.clear()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 删除旧 annotation（如果存在）
        self._annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
            arrowprops=dict(arrowstyle="->"),
        )
        self._annot.set_visible(False)

        # 区间事件
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
            self._lane_durations[lane] += e.get("dur", 0)  # 微秒

        t0 = min(e["ts"] for e in events)

        yticks, ylabels = [], []
        y = 0
        height = 0.6

        for lane_name, evs in list(lanes.items())[:30]:
            for e in evs:
                start = (e["ts"] - t0) / 1000.0
                dur = e["dur"] / 1000.0
                bars = ax.barh(
                    y,
                    dur,
                    left=start,
                    height=height,
                    color=self._color_for(lane_name),
                    alpha=0.8
                )
                rect = bars[0]
                self._bars.append((rect, e, lane_name))

            yticks.append(y)
            ylabels.append(lane_name)
            y += 1

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Time (ms)")
        title = f"{model} on {backend} batch size {batch_size} Inference Timeline" if model else "Inference Timeline"
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        self.figure.subplots_adjust(left=0.25)

        self.canvas.draw_idle()

    def _on_hover(self, event):
        if event.inaxes is None:
            if self._annot and self._annot.get_visible():
                self._annot.set_visible(False)
                self.canvas.draw_idle()
            return

        for rect, e, lane_name in self._bars:
            contains, _ = rect.contains(event)
            if contains:
                # 当前事件和 lane 总时间
                event_ms = e["dur"] / 1000.0
                total_ms = self._lane_durations[lane_name] / 1000.0
                text = f"{lane_name}\nEvent Duration: {event_ms:.3f} ms\nTotal Duration: {total_ms:.3f} ms"

                # 固定 tooltip 在左侧 y 对应 lane 的位置
                y = rect.get_y() + rect.get_height() / 2
                self._annot.xy = (0, y)  # x=0 固定左侧
                self._annot.xycoords = 'data'  # 使用数据坐标
                self._annot.set_text(text)
                self._annot.set_visible(True)
                self.canvas.draw_idle()
                return

        # 没命中任何 bar
        if self._annot.get_visible():
            self._annot.set_visible(False)
            self.canvas.draw_idle()

    @staticmethod
    def _color_for(name: str):
        palette = [
            "#60A5FA", "#34D399", "#FBBF24",
            "#F87171", "#A78BFA", "#FB7185",
            "#2DD4BF", "#F97316"
        ]
        return palette[hash(name) % len(palette)]
