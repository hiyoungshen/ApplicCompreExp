# 应用综合实验程序练习2
> 试卷生成器：从网上资源构建任意年级/科目题库，从题库文件中随机挑选出不同的题目
生成一份试卷，从对应的答案文件中抽取对应的答案构成标准答案。
## 具体的实现
1. 题目分析
* 会读写操作excel,从excel中读取题目随机的生成几张卷子即可。
* 不读写excel的话也可以直接操纵数据库
* 受制于资源限制，使用excel生成（网上找到了excel题库）。
2. 使用函数
* python-docs=0.8.11
* openpyxl=3.0.7
* tkinter(python内置GUI，生成一个简单的小按钮).
3. 实现分析
* 实现了一个类GenerateQuestions，实现生成题目的功能
* 主函数PaperTestGenerator调用GenerateQuestions的类，顺便做个简单的GUI。