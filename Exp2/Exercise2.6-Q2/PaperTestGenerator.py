# coding=utf-8
from tkinter import *
from tkinter.filedialog import *
from GenerateQuestions import GenerateQuestions

if __name__ == "__main__":
    tk = Tk()
    tk.title("PaperTestGenerator")

    mainWindow = Frame(tk, width=235, height=120)
    mainWindow.grid_propagate(0)
    mainWindow.grid()
    
    subWindow = Frame(mainWindow, width=800, height=400)
    subWindow.grid_propagate(0)
    subWindow.grid()

    entry = Entry(subWindow)
    filefound = GenerateQuestions()

    button1 = Button(subWindow, text="选择Excel文件生成", command=filefound.generateQuestions, width=30, height=10).grid(row=0, column=0)
    
    mainloop()