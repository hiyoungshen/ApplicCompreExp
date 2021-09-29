from os import curdir
import requests
from lxml import etree
import json
import csv
import time
import random
import tkinter as tk
import csv
import time
from tkinter import *
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import ttk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import threading

def doCrawlMovieList():
    try:
        scr.insert(INSERT, "Begin crawling...\n")
        url = "https://movie.douban.com/top250?start={page}&filter="
        fd = open("MovieTop250.csv", "w", encoding="utf-8", newline="")
        for page in range(0, 250, 25):
            scr.insert(
                INSERT,
                "Crawling page " + str(page + 1) + " to page " + str(page + 25) + "\n",
            )

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
            }
            response = requests.get(url=url.format(page=str(page)), headers=headers)
            html = response.text

            html_elem = etree.HTML(html)
            links = html_elem.xpath('//div[@class="hd"]/a/@href')
            titles = html_elem.xpath('//div[@class="hd"]/a/span[1]/text()')
            data = zip(links, titles)

            writer = csv.writer(fd)
            for item in data:
                writer.writerow(item)
        fd.close()
        print("End crawling!")
        scr.insert(INSERT, "End crawling! ")
    except EXCEPTION as ex:
        scr.insert(INSERT, "Error!\n")
        tk.messagebox.showerror(title="Hi", message=ex)


"""
爬取Top250的电影信息
"""


def crawlMovieList():
    start = time.time()
    p = threading.Thread(target=doCrawlMovieList)
    p.start()
    stop = time.time()

def doCrawlComment(show_type, i):
    try:
        scr.insert(INSERT, "Begin crawling comment....\n")
        url_base = i + "comments?limit=20&status=P&sort=new_score"
        fd = open("{}.csv".format(show_type), "w", encoding="utf-8", newline="")
        for page in range(0, 200, 20):
            scr.insert(
                INSERT,
                "Crawling comment " + str(page + 1) + " to comment " + str(page + 20) + "\n",
            )
            if page < 20:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
                }
                response = requests.get(url=url_base, headers=headers)
                txt = response.text
            else:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
                }
                response = requests.get(url=url.format(page=page), headers=headers)
                txt = response.text
            
            txt_elem = etree.HTML(txt)
            name = txt_elem.xpath('//div[@class="comment"]/p/span[1]/text()')  # 评论
            data = zip(name)

            writer = csv.writer(fd)
            for item in data:
                writer.writerow(item)

            time.sleep(random.random())
            url = i + "comments?start={page}&limit=20&status=P&sort=new_score"
        fd.close()
        showMovieList(show_type)
        scr.insert(INSERT, "{}. End crawing comments.\n".format(show_type))
    except EXCEPTION as ex:
        scr.insert(INSERT, "Error!\n")
        tk.messagebox.showerror(title="Hi", message=ex)


def crawlComment():
    start = time.time()
    p = threading.Thread(target=doCrawlComment, args=(En.get(), searchByName(En.get())))
    p.start()
    stop = time.time()


def searchByName(show_type):
    with open("MovieTop250.csv", encoding="utf-8") as f:
        data_list = [i for i in csv.reader(f)]
    dict = {}
    for i in data_list:
        dict[i[1]] = i[0]
    return dict.get(show_type)


def showMovieList(show_type):
    master = tk.Tk()
    master.geometry("400x400")
    scr = scrolledtext.ScrolledText(master, width=90, height=80)
    scr.pack()
    if show_type == 1:
        master.title("show movie list")
        with open("MovieTop250.csv", encoding="utf-8") as fp:
            data_list = [i for i in csv.reader(fp)]
        for item in data_list:
            scr.insert(15000.0, item[1] + "\n")
    else:
        master.title("{}show comment list".format(show_type))
        with open("{}.csv".format(show_type), encoding="utf-8") as fp:
            data_list = [i for i in csv.reader(fp)]
        for item in data_list:
            scr.insert(15000.0, item[0] + "\n")
    master.mainloop()


def generateWordCloud(c):
    try:
        
        scr.insert(INSERT, "{}. Begin generating cloud\n".format(c))
        path_txt = "{}.csv".format(c)
        f = open(path_txt, "r", encoding="UTF-8").read()
        stop = {
            "的",
            "我",
            "我们",
            "你们",
            "他们",
            "它",
            "她",
            "他",
            "你",
            "她们",
            "电影",
            "是",
            "还是",
            "了",
            "就",
            "也",
            "和",
            "看",
            "一个",
            "在",
            "都",
            "不",
            "有",
            "没有",
            "那",
            "那个",
            "这",
            "这个",
            "啊",
            "又",
        }
        cut_txt = " ".join(jieba.cut(f))
        wordcloud = WordCloud(
            font_path="./simfang.ttf",
            background_color="white",
            width=1000,
            height=900,
            stopwords=stop,
        ).generate(cut_txt)
        
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        wordcloud.to_file("{}.jpg".format(c))
    except:
        tk.messagebox.showerror(title="Hi", message="Error!")


# 主函数
if __name__ == "__main__":
    window = tk.Tk()
    window.title("Crawle douban film review")
    window.geometry("400x450")
    btn1 = tk.Button(
        window,
        text="Crawle douban top 250 films",
        font=("Arial", 11),
        width=30,
        height=1,
        command=crawlMovieList,
    )
    btn1.place(x=80, y=10)

    btn2 = tk.Button(
        window,
        text="Show info of top 250 films",
        font=("Arial", 11),
        width=30,
        height=1,
        command=lambda: showMovieList(1),
    )
    btn2.place(x=80, y=50)

    l = tk.Label(
        window,
        text="Which film you want to know?",
        font=("Arial", 11),
        width=30,
        height=2,
    )
    l.place(x=80, y=80)

    En = tk.Entry(window, width=40)
    En.place(x=80, y=115)
    btn3 = tk.Button(
        window,
        text="Begin crawling",
        font=("Arial", 12),
        width=30,
        height=1,
        command=crawlComment,
    )
    btn3.place(x=80, y=145)
    btn4 = tk.Button(
        window,
        text="Generating word cloud and save it",
        font=("Arial", 12),
        width=30,
        height=1,
        command=lambda: generateWordCloud(En.get()),
    )
    btn4.place(x=80, y=180)

    scr = scrolledtext.ScrolledText(window, width=54, height=18)
    scr.place(y=220)
    window.mainloop()
