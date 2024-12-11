import sys
import winsound
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QRadioButton, QButtonGroup, QSlider, QWidget, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_face_detection()

    def initUI(self):
        self.setWindowTitle('졸음 감지기 (Drowsiness Detector) ')
        self.setGeometry(100, 100, 600, 400)  # 창 크기 설정

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel('졸음 감지기 (Drowsiness Detector) ')
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; text-align: center; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Video and Control Panel Layout
        video_control_layout = QHBoxLayout()

        # Video Display
        self.video_label = QLabel()
        self.video_label.setFixedSize(650, 380)  # 영상 크기 고정
        self.video_label.setStyleSheet("border: 1px solid black;")
        video_control_layout.addWidget(self.video_label)

        # Control Panel
        control_layout = QVBoxLayout()

        # EAR Sensitivity Settings
       # sensitivity_label = QLabel('민감도 설정 (0.16 ~ 0.23):')

       # sensitivity_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        #control_layout.addWidget(sensitivity_label)

        sensitivity_group = QVBoxLayout()
        self.radio_group = QButtonGroup(self)
        self.sensitivity = 0.18  # Default sensitivity

        for i, value in enumerate(np.linspace(0.16, 0.23, 9)):
            radio_button = QRadioButton(f"{i + 1} 단계")
            radio_button.value = value
            self.radio_group.addButton(radio_button, i)
            if i % 3 == 0:
                row_layout = QHBoxLayout()
                sensitivity_group.addLayout(row_layout)
            row_layout.addWidget(radio_button)

        self.radio_group.buttonClicked.connect(self.update_sensitivity)
        control_layout.addLayout(sensitivity_group)

        # Warning Sound Volume Dial
        #volume_label = QLabel('경고음 조절 (50 ~ 3000 Hz):')
       # volume_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
       # control_layout.addWidget(volume_label)

        self.volume_dial = QSlider(Qt.Horizontal)
        self.volume_dial.setRange(50, 3000)
        self.volume_dial.setValue(2000)  # Default frequency
        control_layout.addWidget(self.volume_dial)

        # Log Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedWidth(250)  # 감지 로그 창 폭 설정
        self.log_area.setFixedHeight(272)  # 영상 창 세로 길이의 4/5로 설정
        self.log_area.setStyleSheet("background-color: #f5f5f5; border: 1px solid gray;")
       # control_layout.addWidget(QLabel('감지 로그:'))
        control_layout.addWidget(self.log_area)

        video_control_layout.addLayout(control_layout)
        main_layout.addLayout(video_control_layout)
        central_widget.setLayout(main_layout)

        # Video Capture Setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms intervals

        self.drowsy_count = 0
        self.max_drowsy_count = 10

    def setup_face_detection(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def update_sensitivity(self, button):
        self.sensitivity = button.value
        self.log_area.append(f"{QTime.currentTime().toString('HH:mm:ss')} - 민감도 설정: {self.sensitivity:.2f}")

    def calculate_ear(self, landmarks, eye_indices):
        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        vertical1 = distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        vertical2 = distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        horizontal = distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

        return (vertical1 + vertical2) / (2 * horizontal)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        is_drowsy = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                             for lm in face_landmarks.landmark]

                left_eye = self.calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
                right_eye = self.calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
                ear = (left_eye + right_eye) / 2.0

                if ear < self.sensitivity:
                    self.drowsy_count += 1
                    is_drowsy = self.drowsy_count > self.max_drowsy_count
                else:
                    self.drowsy_count = max(0, self.drowsy_count - 1)

        if is_drowsy:
            cv2.putText(frame, "DROWSY!!!", (250, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0, 0), 11)
            self.log_area.append(f"{QTime.currentTime().toString('HH:mm:ss')} - 졸음 감지됨!!!!!!")
            winsound.Beep(self.volume_dial.value(), 500)


        # Convert frame to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        self.face_mesh.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    detector = DrowsinessDetector()
    detector.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()