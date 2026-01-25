from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QSizePolicy
)
from PyQt5.QtCore import Qt

from gui.ui.pages.inference_page import InferencePage
from gui.ui.pages.ir_page import IRPage
from gui.ui.pages.performance_page import PerformancePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("异构GPU推理中间件原型系统")
        self.resize(1400, 900)

        # ====================
        # 主容器
        # ====================
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ====================
        # 左侧菜单栏
        # ====================
        self.menu_widget = QWidget()
        self.menu_widget.setFixedWidth(200)

        menu_layout = QVBoxLayout(self.menu_widget)
        menu_layout.setContentsMargins(0, 20, 0, 20)
        menu_layout.setSpacing(10)

        self.btn_inference = QPushButton("推理控制")
        self.btn_ir = QPushButton("IR 可视化")
        self.btn_perf = QPushButton("性能指标")

        for btn in [self.btn_inference, self.btn_ir, self.btn_perf]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFixedHeight(44)
            btn.setCheckable(True)
            menu_layout.addWidget(btn)

        menu_layout.addStretch()
        main_layout.addWidget(self.menu_widget)

        # style控制
        self.menu_widget.setObjectName("SideBar")

        self.btn_inference.setObjectName("SideBarButton")
        self.btn_ir.setObjectName("SideBarButton")
        self.btn_perf.setObjectName("SideBarButton")

        # 页面
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        self.inference_page = InferencePage()
        self.ir_page = IRPage()
        self.performance_page = PerformancePage()

        self.stack.addWidget(self.inference_page)    # index 0
        self.stack.addWidget(self.ir_page)           # index 1
        self.stack.addWidget(self.performance_page)  # index 2

        # 逻辑连接
        self.btn_inference.clicked.connect(lambda: self.switch_page(0))
        self.btn_ir.clicked.connect(lambda: self.switch_page(1))
        self.btn_perf.clicked.connect(lambda: self.switch_page(2))

        # 默认页
        self.switch_page(0)

    # 页面切换
    def switch_page(self, index: int):
        self.stack.setCurrentIndex(index)

        # 同步侧边栏高亮
        for i, btn in enumerate(
            [self.btn_inference, self.btn_ir, self.btn_perf]
        ):
            btn.setChecked(i == index)
