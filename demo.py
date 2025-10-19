import sys
import os
import argparse

#sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QUrl, QEvent, QFileInfo
# from PyQt5 import QtMultimedia
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from util.image_warp import crop2_169, resize_img

from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor

ROTATION_MAPPING = {
    "90": cv2.ROTATE_90_CLOCKWISE,
    "-90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
    "0": None,
}

class VitonThread(QThread):
    frameCaptured = pyqtSignal(QImage)

    def __init__(self, rotation_setting=None):
        super().__init__()
        self.rotation_setting = rotation_setting
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.rotate_code = self._get_rotation_code()
        self.running = True
        self.frame_processor = FullBodySeqFrameProcessor('coat_seq_vmssdp2ta_576')
        #self.frame_processor = FullBodyFrameProcessor('han_baseline_vmsdp2ta_576')
        self.use_vmssdp = False



    def run(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                if self.rotate_code is not None:
                    frame = cv2.rotate(frame, self.rotate_code)
                ## ichao: remove flip (Nov 13, 2024)
                frame=cv2.flip(frame, 1)
                frame=resize_img(frame,max_height=1024)
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame=crop2_169(frame)

                frame = self.frame_processor.forward(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.frameCaptured.emit(q_img)


    def stop(self):
        self.running = False
        self.cap.release()

    def _get_rotation_code(self):
        if self.rotation_setting is not None:
            rotation_key = str(self.rotation_setting).strip()
            if rotation_key in ROTATION_MAPPING:
                print(f"Applying rotation override {rotation_key}° from CLI.")
                return ROTATION_MAPPING[rotation_key]
            print(f"Unsupported CLI rotation value: {self.rotation_setting}. Expected one of {list(ROTATION_MAPPING.keys())}.")

        env_val = os.getenv("RTV_CAMERA_ROTATE")
        if env_val is not None:
            env_val = env_val.strip()
            if env_val in ROTATION_MAPPING:
                print(f"Applying rotation {env_val}° from RTV_CAMERA_ROTATE.")
                return ROTATION_MAPPING[env_val]
            print(f"Unsupported RTV_CAMERA_ROTATE value: {env_val}. Expected one of {list(ROTATION_MAPPING.keys())}.")
        print("RTV_CAMERA_ROTATE not set; using camera stream orientation as-is.")
        return None

class CameraApp(QMainWindow):
    def __init__(self, rotation_setting=None):
        super().__init__()
        self.rotation_setting = rotation_setting
        self.setWindowTitle("Virtual Try-On")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(200, 150)
        ## enable fullscreen
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)

        # Create a scroll area for the horizontal layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a QWidget to hold the list

        #layout.setStretch(0, 9)
        #layout.setStretch(1, 2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.viton_thread = VitonThread(rotation_setting=self.rotation_setting)
        self.viton_thread.frameCaptured.connect(self.update_image)
        self.viton_thread.start()


    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.viton_thread.stop()
        self.viton_thread.wait()
        print("Viton thread stopped")
        event.accept()

def parse_cli_args():
    parser = argparse.ArgumentParser(description="RTV demo")
    parser.add_argument(
        "--camera-rotate",
        choices=list(ROTATION_MAPPING.keys()),
        help="Rotate camera feed by given degrees (overrides RTV_CAMERA_ROTATE).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    args = parse_cli_args()
    app = QApplication(sys.argv)
    window = CameraApp(rotation_setting=args.camera_rotate)
    window.show()
    sys.exit(app.exec_())
