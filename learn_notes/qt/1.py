import sys
from PyQt5.QtWidgets import QWidget,QPushButton,QGridLayout, QLineEdit, QTextEdit,QHBoxLayout, QVBoxLayout, QLabel, QMessageBox,QDesktopWidget, QApplication


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        #窗口的右下角显示两个按钮
        # okButton = QPushButton("OK")
        # cancelButton = QPushButton("Cancel")
        #
        # hbox = QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(okButton)
        # hbox.addWidget(cancelButton)
        #
        # vbox = QVBoxLayout()
        # vbox.addStretch(1)
        # vbox.addLayout(hbox)

        #self.setLayout(vbox)

        #内容

        title = QLabel('Title')
        author = QLabel('Author')
        review = QLabel('Review')

        titleEdit = QLineEdit()
        authorEdit = QLineEdit()
        reviewEdit = QTextEdit()

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(title, 1, 0)
        grid.addWidget(titleEdit, 1, 1)

        grid.addWidget(author, 2, 0)
        grid.addWidget(authorEdit, 2, 1)

        grid.addWidget(review, 3, 0)
        grid.addWidget(reviewEdit, 3, 1, 5, 1)

        self.setLayout(grid)


        # lbl1 = QLabel('Zetcode', self)
        # lbl1.move(15, 10)
        #
        # lbl2 = QLabel('tutorials', self)
        # lbl2.move(35, 40)
        #
        # lbl3 = QLabel('for programmers', self)
        # lbl3.move(55, 70)
        ##大小居中
        self.setGeometry(300, 300, 350, 300)
        self.center()
        self.setWindowTitle('Message box')
        self.show()

    def center(self):

        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())