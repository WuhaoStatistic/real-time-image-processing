from gui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from generators import ResnetGenerator
import torch
import numpy as np
from torchvision import transforms


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.timer_camera = QtCore.QTimer()  # update screen
        self.cap = cv2.VideoCapture()  # video stream
        self.CAM_NUM = 0  # 0 means carema
        self.processing = False
        self.img = None
        self.model = get_generator('nature_photo.pth')
        self.brightness.setMaximum(200)
        self.saturation.setMaximum(200)
        self.brightness.setValue(100)
        self.saturation.setValue(100)
        self.b = 0
        self.s = 0
        # slots
        self.open.clicked.connect(self.open_camera_clicked)
        self.start.clicked.connect(self.show_process)
        self.end.clicked.connect(self.close)
        self.timer_camera.timeout.connect(self.show_camera)
        self.cart.toggled.connect(lambda: self.ratio_set_function(btn=self.cart))
        self.paint.toggled.connect(lambda: self.ratio_set_function(btn=self.paint))
        self.brightness.valueChanged.connect(self.do_slider)
        self.saturation.valueChanged.connect(self.do_slider)
        #
        self.cart.setChecked(True)
        self.algorithm = 'cart'
        self.label.setVisible(True)
        self.label_2.setVisible(True)
        self.brightness.setVisible(True)
        self.saturation.setVisible(True)

    def do_slider(self):
        self.b = float(self.brightness.value() - 100) / 100.0
        self.s = float(self.saturation.value() - 100) / 100.0

    def ratio_set_function(self, btn):
        if btn.text() == 'Cartoonize':
            if btn.isChecked():
                self.algorithm = 'cart'
                self.label.setVisible(True)
                self.label_2.setVisible(True)
                self.brightness.setVisible(True)
                self.saturation.setVisible(True)

        if btn.text() == 'Painting-style':
            if btn.isChecked():
                self.algorithm = 'cycle'
                self.label.setVisible(False)
                self.label_2.setVisible(False)
                self.brightness.setVisible(False)
                self.saturation.setVisible(False)

    def open_camera_clicked(self):
        if not self.timer_camera.isActive():  # ?????????????????????
            flag = self.cap.open(self.CAM_NUM)  # ?????????0???????????????????????????????????????????????????????????????????????????????????????
            if not flag:  # flag??????open()????????????
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "??????????????????????????????????????????",
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(50)  # ?????????????????????30ms??????????????????50ms??????????????????????????????
                self.open.setText('close camera')
        else:
            self.timer_camera.stop()  # ???????????????
            self.cap.release()  # ???????????????
            self.original.clear()  # ????????????????????????
            self.open.setText('open camera')
            self.processing = False
            self.process.clear()  # ????????????????????????
            self.start.setText('start')

    def show_camera(self):
        flag, self.img = self.cap.read()  # ?????????????????????

        show = cv2.resize(self.img, (768, 768))  #
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  #
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # ?????????????????????????????????QImage??????
        self.original.setPixmap(QtGui.QPixmap.fromImage(showImage))  # ??????????????????Label??? ??????QImage
        img = show
        if self.processing:
            if self.algorithm == 'cycle':
                img = pre_process(img).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    img = (self.model.forward(img)).cpu().numpy()
                    img = ((np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
                    img = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2]])

            if self.algorithm == 'cart':
                img = get_cartoon(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), l=self.b, s=self.s)
                img = cv2.resize(img, (768, 768))
                img = get_cartoon(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            img = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                               QtGui.QImage.Format_RGB888)
            self.process.setPixmap(QtGui.QPixmap.fromImage(img))

    def show_process(self):
        if not self.processing:
            self.processing = True
            self.start.setText('stop process')
        else:
            self.processing = False
            self.process.clear()  # ????????????????????????
            self.start.setText('start')
            self.label.setVisible(False)
            self.label_2.setVisible(False)
            self.brightness.setVisible(False)
            self.saturation.setVisible(False)


def get_generator(path):
    g = ResnetGenerator(3, 3)
    g.load_state_dict(torch.load(path))
    g.to('cuda' if torch.cuda.is_available() else 'cpu')
    g.eval()
    return g


def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    # transform_list += [transforms.Resize(768, interpolation=transforms.InterpolationMode.BICUBIC)]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def pre_process(img):
    trans = get_transform()
    img = trans(img)

    return img


def get_cartoon(p, l=0.1, s=0.95):
    p = cv2.bilateralFilter(p, 5, 150, 150)

    gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    p2 = cv2.Canny(gray, 30, 100)

    p = p.astype(np.float32) / 255.0
    p = cv2.cvtColor(p, cv2.COLOR_BGR2HLS)

    p[:, :, 1] = (1.0 + l) * p[:, :, 1]
    p[:, :, 1][p[:, :, 1] > 1] = 1
    # ?????????
    p[:, :, 2] = (1.0 + s) * p[:, :, 2]
    p[:, :, 2][p[:, :, 2] > 1] = 1
    # HLS2BGR
    p = cv2.cvtColor(p, cv2.COLOR_HLS2BGR) * 255
    p = np.array(p, dtype=np.int16)
    p[:, :, 0] = np.clip(p[:, :, 0] - p2, 0, 255)
    p[:, :, 1] = np.clip(p[:, :, 1] - p2, 0, 255)
    p[:, :, 2] = np.clip(p[:, :, 2] - p2, 0, 255)
    p = np.array(p, dtype=np.uint8)
    p = cv2.merge([p[:, :, 0], p[:, :, 1], p[:, :, 2]])
    return p
