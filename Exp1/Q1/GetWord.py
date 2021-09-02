# # -*- coding: UTF-8 -*-
# import requests
# import re
# from bs4 import BeautifulSoup

# # 下载一个网页
# url = 'https://blog.reedsy.com/short-story/dol81e/'

# # 模拟浏览器发送http请求
# response = requests.get(url)
# # response.encoding='utf-8'
# response.encoding=response.apparent_encoding

# html = BeautifulSoup(response.text,'lxml')

# content=html.select('body > div.writing-prompts > section.row-thin.row-white > div > article')
# content=html.select('body > div.writing-prompts > section.row-thin.row-white > div > article > p:nth-child(2)')

# # 目标小说主页的网页源码
# # html = response.text
# print(content)


import requests
#导入正则表达式
import re
# 下载一个网页
url = 'https://read.qidian.com/chapter/J2XED_z8FNnxq9ZHzk0vMw2/Nr6SNV6L0QT6ItTi_ILQ7A2/'
# 模拟浏览器发送http请求
response = requests.get(url)
# 编码方式
# response.encoding = 'utf-8'
response.encoding=response.apparent_encoding
# 目标小说主页的网页源码
html = response.text
# print(html)
# 小说的名字
# title = re.findall(r'<meta property="og:title" content="(.*?)"\>',html)[0]

title = "我被迫挖了邪神的墙脚"

# 新建一个文件，保存小说内容(以写的方式）
fb = open('%s.txt' % title, 'w', encoding='utf-8')

# 获取每一章的信息（章节，url）
chapters_response = requests.get('https://book.qidian.com/info/1029884110/#Catalog')
chapters=chapters_response.text
# print(chapters)
# <li data-rid="1"><a href="//read.qidian.com/chapter/J2XED_z8FNnxq9ZHzk0vMw2/Nr6SNV6L0QT6ItTi_ILQ7A2/" target="_blank" data-eid="qd_G55" data-cid="//read.qidian.com/chapter/J2XED_z8FNnxq9ZHzk0vMw2/Nr6SNV6L0QT6ItTi_ILQ7A2/" title="首发时间：2021-08-02 12:02:30 章节字数：1099">第一章 异草</a>
chapter_info_list = re.findall(r'<a href="//(.*?)" title="首发时间(.*?)">(.*?)</a>',chapters)
# print(type(chapter_info_list[0]))
# print(chapter_info_list[0])
chapter_info_list = [(cur_chapter[0].split('" target=')[0], cur_chapter[-1]) for cur_chapter in chapter_info_list]
# chapter_info_list = re.findall(r'h<a href="(.*?)" target="(.*?)" data-eid="(.*?)" data-cid=""(.*?)" title="(.*?)">(.*?)</a>',chapters)
# f=open("test.txt","w")
# f.write(chapters)
# f.close()
print(chapter_info_list[10])
# # chapter_info_list = re.findall(r'href="(.*?)">(.*?)<',dl)
# 循环每一个章节，分别去下载
for chapter_info in chapter_info_list:
    chapter_url,chapter_title = chapter_info
    # 下载章节的内容
    chapter_url="http://"+chapter_url
    chapter_response = requests.get(chapter_url)
    chapter_response.encoding = 'utf-8'
    chapter_html = chapter_response.text
    # 提取章节内容
    # chapter_content = re.findall(r'<script>a1\(\);</script>(.*?)<script>a2\(\);</script>',chapter_html, re.S)[0]
    chapter_content = re.findall(r'<div class="read-content j_readContent" id="">(.*?)</div>',chapter_html, re.S)[0]
    # print(chapter_content)
    # input()

    # # 清洗数据,替换空格，换行符
    chapter_content = chapter_content.replace(' ','')
    chapter_content = chapter_content.replace('\t','')
    chapter_content = chapter_content.replace('&nbsp','')
    chapter_content = chapter_content.replace('<br/>','')
    chapter_content = chapter_content.replace('<br>', '')
    chapter_content = chapter_content.replace('<p>', '')
    chapter_content = chapter_content.replace('　', '')
    #持久化,写入标题，内容并换行
    fb.write(chapter_title)
    fb.write(chapter_content)
    fb.write('\n')
    print(chapter_url)
fb.close()