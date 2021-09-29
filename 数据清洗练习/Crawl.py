# coding=utf-8
"""根据搜索词下载百度图片"""
import re
from ssl import VerifyFlags
import sys
import urllib
from pygments.lexer import words
 
import requests
 
 
def get_onepage_urls(onepageurl):
    """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
    if not onepageurl:
        print('已到最后一页, 结束')
        return [], ''
    try:
        headers = {"User-Agent" : "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
        html = requests.get(onepageurl, headers=headers, allow_redirects=False, verify=False).text
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    # print("here: ",html)
    pic_urls = re.findall('"thumbURL":"(.*?)",', html, re.S)
    fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    # print(pic_urls)
    # print(len(pic_urls))
    return pic_urls, fanye_url
 
 
def down_pic(pic_urls):
    """给出图片链接列表, 下载所有图片"""
    for i, pic_url in enumerate(pic_urls):
        try:
            print(pic_url)
            # input()
            pic = requests.get(pic_url, timeout=15)
            string = "./pic_save/"+str(i + 1) + '.jpg'
            with open(string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue
 
 
if __name__ == '__main__':
    """
    https://image.baidu.com/search/acjson?tn=resultjson_com&logid=6537233715412755468&ipn=rj&ct=201326592&is=&fp=result&queryWord=猫&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word=猫&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&nojc=&pn=30&rn=30&gsm=1e&1632387099279=
    https://image.baidu.com/search/acjson?tn=resultjson_com&logid=6537233715412755468&ipn=rj&ct=201326592&is=&fp=result&queryWord=猫&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word=猫&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&nojc=&pn=60&rn=30&gsm=3c&1632387100233=
    """
    
    keyword = '猫'  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
    url_init_first = r'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=7410591006154034041&ipn=rj&ct=201326592&is=&fp=result&queryWord=猫&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word={word}&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&nojc=&pn={pn}&rn={rn}&gsm={gsm}&'
    
    word=urllib.parse.quote(keyword, safe='/')
    pn=30
    rn=30
    gsm=hex(pn)[2:]
    
    url_init = url_init_first.format(word=word,pn=pn, rn=rn, gsm=gsm)
    all_pic_urls = []
    onepage_urls, fanye_url = get_onepage_urls(url_init)
    all_pic_urls.extend(onepage_urls)
 
    fanye_count = 0  # 累计翻页数

    num=1
    while 1:
        if num>20:
            break
        num+=1

        pn+=30
        url_init = url_init_first.format(word=word, pn=pn+30, rn=rn, gsm=hex(pn)[2:])
        onepage_urls, fanye_url = get_onepage_urls(url_init)

        fanye_count += 1
        print('第%s页' % fanye_count)
        all_pic_urls.extend(onepage_urls)
 
    down_pic(list(set(all_pic_urls)))