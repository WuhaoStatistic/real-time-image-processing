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
        # slots
        self.open.clicked.connect(self.open_camera_clicked)
        self.start.clicked.connect(self.show_process)
        self.end.clicked.connect(self.close)
        self.timer_camera.timeout.connect(self.show_camera)

    def open_camera_clicked(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if not flag:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(50)  # 定时器开始计时30ms，结果是每过50ms从摄像头中取一帧显示
                self.open.setText('close camera')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.original.clear()  # 清空视频显示区域
            self.open.setText('open camera')
            self.processing = False
            self.process.clear()  # 清空视频显示区域
            self.start.setText('start')

    def show_camera(self):
        flag, self.img = self.cap.read()  # 从视频流中读取

        show = cv2.resize(self.img, (768, 768))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.original.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        if self.processing:
            img = pre_process(show).to('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                img = (self.model.forward(img)).cpu().numpy()
                img = ((np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
                img = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
                img = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                   QtGui.QImage.Format_RGB888)
                self.process.setPixmap(QtGui.QPixmap.fromImage(img))

    def show_process(self):
        if not self.processing:
            self.processing = True
            self.start.setText('stop process')
        else:
            self.processing = False
            self.process.clear()  # 清空视频显示区域
            self.start.setText('start')


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
