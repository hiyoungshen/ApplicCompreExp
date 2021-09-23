"""
数据集的采集
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import string
from collections import OrderedDict

# 剔除单字符的“单词”，除非这个字符是“i”或“a”；
# 剔除维基百科的引用标记（方括号包裹的数字，如 [1]）；
# 剔除标点符号。
def cleanInput(input):
    input = re.sub('\n+', " ", input)
    input = re.sub('\[[0-9]*\]', "", input)
    input = re.sub(' +', " ", input)
    input = bytes(input, "UTF-8")
    input = input.decode("ascii", "ignore")
    cleanInput = []
    input = input.split(' ')
    for item in input:
        item = item.strip(string.punctuation)
        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
            cleanInput.append(item)
    return cleanInput

def getNgrams(input, n):
    input = cleanInput(input)
    output = dict()
    for i in range(len(input)-n+1):
        newNGram = " ".join(input[i:i+n])
        if newNGram in output:
            output[newNGram] += 1
        else:
            output[newNGram] = 1
    return output

# 首先用一些正则表达式来移除转义字符（ \n ），再把 Unicode 字符过滤掉。
html = urlopen("https://baike.baidu.com/item/Python/407313")
bsObj = BeautifulSoup(html, "html.parser")
content = bsObj.find("div", {"class":"main-content"}).get_text()
# print(content)
ngrams = getNgrams(content, 2)


ngrams = OrderedDict(sorted(ngrams.items(), key=lambda t: t[1], reverse=True))
print(f"数据量为{len(ngrams)}")
