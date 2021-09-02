# Question 1: 统计文本文件中各字母（区分大小写）的出现频率
import os

f = open("test.txt", 'r', encoding='UTF-8')
lines = f.readlines()
l = ''

characters = []
freq = {}

for line in lines:
    line = line.strip()
    if len(line) == 0:
        continue
    l = l + line
    
    for x in range(0, len(line)):
        if not line[x] in characters:
            characters.append(line[x])
            
        if line[x] not in freq:
            freq[line[x]] = 1
            
        freq[line[x]] += 1

freq = sorted(freq.items(), key = lambda e:e[0], reverse = False) # 按照键进行排序
# freq = sorted(freq.items(), key = lambda e:e[1], reverse = True) # 按照值进行排序

for i in freq:
    print("[", i[0], "] 共出现", i[1], "次")

f.close()    