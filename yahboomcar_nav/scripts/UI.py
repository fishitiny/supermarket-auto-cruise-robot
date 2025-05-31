#!/usr/bin/python3
# -*-coding: UTF-8 -*
from tkinter import *
import rospy
from std_msgs.msg import Int32,Bool,String
#from ttkbootstrap import *


class MainWindows(Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Navigator")  # 给界面添加一个标题
        self.geometry("544x344+400+200")  # 定义界面尺寸
        self.pub = rospy.Publisher('mode', Int32, queue_size=10)
        #self.pub = rospy.Publisher('mode', Int32, queue_size=10)
        # self.resizable(0, 0)  # 定义界面窗口大小不可改变

        # 调用常用变量
        self.setup_main_gui()


    def setup_main_gui(self):
        # 创建一个界面标题
        # self.label_title = Label(self, text="test")
        # self.label_title.place(relwidth=1, relheight=0.18, relx=0, rely=0)
        self.setup_frame01()
        # 创建左侧按钮显示区域


    def mode0(self):
        print("mode 0")
        mode = Int32()
        mode.data = 0
        self.pub.publish(mode)

    def mode2(self):
        print("mode 2")
        mode = Int32()
        mode.data = 2
        self.pub.publish(mode)

    def setup_frame01(self):
        self.frame01 = Frame(self, relief="groove")
        self.frame01.place(relwidth=0.84, relheight=0.82, relx=0.16, rely=0.18)
        self.lb = Label(self, text='Pick your search mode', fg='blue')
        self.modeButton0 = Button(self, text="Picture", command=self.mode0)
        self.modeButton2 = Button(self, text="Text", command=self.mode2)
        self.modeButton0.pack()
        self.modeButton2.pack()
        # self.label_01 = Label(self.frame01, text="界面显示一", font="微软雅黑 12 bold", bg="green")
        # self.label_01.place(relwidth=1, relheight=1, relx=0, rely=0)

    def setup_frame02(self):
        self.frame02 = Frame(self, relief="groove")
        self.frame02.place(relwidth=0.84, relheight=0.82, relx=0.16, rely=0.18)
        self.label_02 = Label(self.frame02, text="界面显示二", bg="red")
        self.label_02.place(relwidth=1, relheight=1, relx=0, rely=0)

    def setup_frame03(self):
        self.frame03 = Frame(self, relief="groove")
        self.frame03.place(relwidth=0.84, relheight=0.82, relx=0.16, rely=0.18)
        self.label_03 = Label(self.frame03, text="界面显示三", bg="blue")
        self.label_03.place(relwidth=1, relheight=1, relx=0, rely=0)

    def createframe01(self):
        try:
            self.frame01.destroy()
        except:
            pass
        finally:
            try:
                self.frame02.destroy()
            except:
                pass
            finally:
                try:
                    self.frame03.destroy()
                except:
                    pass
                finally:
                    self.setup_frame01()

    def createframe02(self):
        try:
            self.frame01.destroy()
        except:
            pass
        finally:
            try:
                self.frame02.destroy()
            except:
                pass
            finally:
                try:
                    self.frame03.destroy()
                except:
                    pass
                finally:
                    self.setup_frame02()

    def createframe03(self):
        try:
            self.frame01.destroy()
        except:
            pass
        finally:
            try:
                self.frame02.destroy()
            except:
                pass
            finally:
                try:
                    self.frame03.destroy()
                except:
                    pass
                finally:
                    self.setup_frame03()


if __name__ == "__main__":
    rospy.init_node('UI', anonymous=False)
    windows = MainWindows()
    windows.mainloop()
    

