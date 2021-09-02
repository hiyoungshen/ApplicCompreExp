# Question 1: 统计文件中各单词（区分大小写）的出现频率，
#             需要注意的是，要移除标点符号。
import os

f = open("test.txt", 'r', encoding='UTF-8')
l = f.read()

# 移除标点符号
for ch in ',*.':
    l = l.replace(ch, " ")

words = l.split()
freq = {}

for w in words:
    freq[w] = freq.get(w, 0) + 1

print(freq)