# Question 2: 试卷生成器：
#             从网上资源构建任意年级/科目题库，从题库文件中随机挑选出不同的题目
#             生成一份试卷，从对应的答案文件中抽取对应的答案构成标准答案.
import os
import random
import json
    
g = os.walk(r"C:\Users\kunkun\Desktop\大四上学期\系统综合实验\Code\第一次实验\2.6.2\problems")
database = []

for path, dir_list, file_list in g:
    for file_name in file_list:
        database.append(os.path.join(path, file_name))
    

# Step 1:必要变量设置    
quest_num = 3 # 抽取题目的数量
question_list = random.sample(database, quest_num)
print(question_list)


# Step 2:从中抽取题目和相应的答案，得到试卷
with open("paper.txt",'w') as F1:
    for i in question_list:
        f =  open(i,'r', encoding='UTF-8')
        question = json.load(f)
        questions = question['article']
        F1.write(questions)
        F1.write("\n\n")
        
    F1.close()


# Step 3:生成标准答案
with open("answer.txt",'w') as F2:
    for i in question_list:
        # ans = ""
        f =  open(i,'r', encoding='UTF-8')
        question = json.load(f)
        F2.write("".join(question['answers']))
        F2.write("\n")
        
    F2.close()