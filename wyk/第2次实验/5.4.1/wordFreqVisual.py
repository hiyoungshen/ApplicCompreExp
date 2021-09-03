# Question1: 可视化展示文件中各单词（区分大小写）的出现频率等信息

# Step 1: 统计各单词的出现频率
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

# print(freq)


# Step 2: 可视化
import matplotlib.pyplot as plt
x = list(freq.keys())
y = list(freq.values())
print(x)
print(y)
plt.figure(figsize=(8,4))
plt.plot(x,y,'o',label="$Freq$", color="pink", linewidth=2)
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title("Word Frequency Statistics")
plt.legend()
plt.show()