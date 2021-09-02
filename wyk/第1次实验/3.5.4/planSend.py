# Question 4: 1)发送邮件；
#             2)发送短信；

# Step 1: Send Email
##        1.1 输入python -m smtpd -c DebuggingServer -n localhost:1025
##        1.2 完成Python程序编写
# import smtplib, ssl

# port = 465  # For SSL
# smtp_server = "smtp.163.com"
# sender_email = "yikunwang6@163.com"  # Enter your address
# receiver_email = "yikunwang6@163.com"  # Enter receiver address
# password = input("请输入邮箱登录密码：") # VXIHWACTZUNCVKKV
# message = """\
# Subject: Yikun's Weekly Schedule

# Wang Yikun, here is your weekly schedule, please CHECK it out and STRICTLY perform these tasks...
#         ------------------------------------------------
#         | 00:00AM -- 06:00AM      | Sleep               |
#         | 06:00AM -- 06:30AM      | Wake Up & Make Up   |
#         | 06:30AM -- 07:10AM      | Morning Exercise    |
#         | 07:10AM -- 07:40AM      | Breakfast           |
#         | 07:40AM -- 12:00AM      | Take Classes        |
#         | 12:00AM -- 00:30PM      | Lunch               |   
#         | 00:30PM -- 05:30PM      | Take Classes        |
#         | 05:30PM -- 06:00PM      | Dinner              |
#         | 06:00PM -- 11:55PM      | Self-Study          |
#         ------------------------------------------------
# Love U!"""

# context = ssl.create_default_context()
# with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#     server.login(sender_email, password)
#     server.sendmail(sender_email, receiver_email, message)

# 参考教程：https://zhuanlan.zhihu.com/p/57252226

# ===================================================================================
# Step 2: Send Message
import requests
 
def send_message(event_name, key, text):
    url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
    payload = "{\n    \"value1\": \""+text+"\"\n}"
    headers = {
        'Content-Type': "application/json",
        'User-Agent': "PostmanRuntime/7.15.0",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,d376ec80-54e1-450a-8215-952ea91b01dd",
        'Host': "maker.ifttt.com",
        'accept-encoding': "gzip, deflate",
        'content-length': "63",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
        }
 
    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
    print(response.text)
 
# text = "叮叮叮，王翊堃同学的每日规划来啦，请查收~"
text = "This is Yikun's schedule."
send_message('send_message', 'o004I3lk8ZGdLWjw9XwnBCERdf7TBajDtk5trI9ObV9', text)

# 参考教程：https://www.bilibili.com/read/cv4062198