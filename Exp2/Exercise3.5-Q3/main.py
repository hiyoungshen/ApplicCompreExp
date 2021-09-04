#!/usr/bin/env python
 
import imaplib
import re
 
def LoginMail(hostname, user, password):
    r = imaplib.IMAP4_SSL(hostname)
    r.login(user, password)
    x, y = r.status('INBOX', '(MESSAGES UNSEEN)')
    r.logout()
    return y[0]
 
 
if __name__ == '__main__':
    # 输入帐号信息，登录163邮箱，也可以选择别的邮箱,可选协议为IMAP，检查该帐号的未读消息
    print(LoginMail('imap.163.com', '123@163.com', '123'))