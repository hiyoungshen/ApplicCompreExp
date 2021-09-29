# import os

# f = open("News.txt", 'r', encoding='UTF-8')
# l = f.read()

# # 移除标点符号
# for ch in '-,*.\n':
#     l = l.replace(ch, " ")

# words = l.split()
# freq = {}

# for w in words:
#     freq[w] = freq.get(w, 0) + 1

# print(freq)
"""
统计单词的个数, 并绘制，柱状图，散点图，折线图
由于单词太多，绘制出的单词的数目过多
"""
import numpy as np
import re
nums={}

with open("News.txt", "r") as f:
    lines=f.readlines()

for line in lines:
    words = line.strip().split()
    for word in words:
        old_word=re.findall("[a-zA-Z]+", word)
        word=""
        for w in old_word:
            word+=w
        if word=="":
            continue
        if word in nums:
            nums[word]+=1
        else:
            nums[word]=1

print(len(nums))

write_name="字频.txt"
write_f=open(write_name, "w")
write_content=""
for name, num in nums.items():
    write_content+=name+"\t"+str(num)+"\n"
write_f.write(write_content)
write_f.close()

names, values = [], []
for name, value in nums.items():
    # print(key, value)
    names.append(name)
    values.append(value)
# 随机消失一些名字
# names = [name if i%20==0 else '' for i, name in enumerate(names)]

import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 3))
x_major_locator = plt.MultipleLocator(1000)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(10)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
plt.subplot(121)
plt.xlabel("word")
plt.ylabel("times")
plt.bar(names[:10], values[:10])
plt.subplot(122)
plt.scatter(names[:10], values[:10])
# plt.subplot(133)
# plt.plot(names, values)
# plt.suptitle('Categorical Plotting')


plt.savefig("Statistic.png")
plt.show()