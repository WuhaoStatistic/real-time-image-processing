from PyQt5.QtWidgets import QApplication
import sys
import numpy as np
from gui_function import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    np.array([2,-1,1,0],ndmin=2)
    
    sys.exit(app.exec_())
