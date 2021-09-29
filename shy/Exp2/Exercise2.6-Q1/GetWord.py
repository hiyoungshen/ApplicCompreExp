"""
在小说网站上下载<<我被迫挖了邪神的墙脚>>的所有内容
"""

import requests
import re

title = "我被迫挖了邪神的墙脚"
fb = open('%s.txt' % title, 'w', encoding='utf-8')

# 在改小说目录获取每一章的信息（章节，url）
chapters_response = requests.get('https://book.qidian.com/info/1029884110/#Catalog')
chapters=chapters_response.text
chapter_info_list = re.findall(r'<a href="//(.*?)" title="首发时间(.*?)">(.*?)</a>',chapters)
chapter_info_list = [(cur_chapter[0].split('" target=')[0], cur_chapter[-1]) for cur_chapter in chapter_info_list]
print(chapter_info_list[10])

# 循环每一个章节，分别去下载
for chapter_info in chapter_info_list:
    chapter_url,chapter_title = chapter_info
    chapter_url="http://"+chapter_url
    chapter_response = requests.get(chapter_url)
    chapter_response.encoding = 'utf-8'
    chapter_html = chapter_response.text
    # 提取章节内容
    chapter_content = re.findall(r'<div class="read-content j_readContent" id="">(.*?)</div>',chapter_html, re.S)[0]

    # 清洗数据,替换空格，换行符
    chapter_content = chapter_content.replace(' ','')
    chapter_content = chapter_content.replace('\t','')
    chapter_content = chapter_content.replace('&nbsp','')
    chapter_content = chapter_content.replace('<br/>','')
    chapter_content = chapter_content.replace('<br>', '')
    chapter_content = chapter_content.replace('<p>', '')
    chapter_content = chapter_content.replace('　', '')
    # 持久化,写入标题，内容并换行
    fb.write(chapter_title)
    fb.write(chapter_content)
    fb.write('\n')
    print(chapter_url)

fb.close()