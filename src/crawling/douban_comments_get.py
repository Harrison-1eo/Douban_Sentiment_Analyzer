
import requests
from bs4 import BeautifulSoup

def get_comments_by_id(id, num):
    """
    通过电影id获取电影的评论
    :param id: 电影id
    :param num: 电影评论的数量
    :return: 电影评论
    """
    url = f"https://movie.douban.com/subject/{id}/comments?start=0&limit={num}&sort=new_score&status=P"
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    }
    proxies = {
    # 'http': '172.30.6.14:8880',
    # 'http': "36.152.44.96:5000",          # 百度ip
    # "http": '111.30.144.71:5000',         # QQip
    # 'http': '47.103.24.173:5000',         # b站ip
    'http': ' 192.168.210.53:5000',       # 本地ip
    # 'http': '60.205.172.2:5000',          # CSDNip
    # 'http': "49.233.242.15:5000",  # 豆瓣ip（doge，想不到吧）
    # 'http': '120.233.21.30:5000',         # 微信ip
    # 'http': '103.41.167.234:5000',        # 知乎ip
    # 'http': '39.156.68.154:5000',         # hao123ip
    # 'http': '36.152.218.86:5000',         # 凤凰网ip
    # 'http': '151.101.78.137:5000',        # 人民网ip
    # 'http': '221.178.37.218:5000',        # 中国网ip
    }

    response = requests.get(url, headers=headers, proxies=proxies)
    response.encoding = 'utf-8'

    # 如果显示“有异常请求从你的 IP 发出”，则说明被封了，需要更换代理ip
    if "有异常请求从你的 IP 发出" in response.text:
        return "IP"

    soup = BeautifulSoup(response.text, 'html.parser')

    comments = soup.select("#comments")
    try:
        comments_ = comments[0].find_all("span", class_='short')
    except IndexError:
        return None

    # 存放最终一条条评论
    final_comments = []

    for comment in comments_:
        final_comments.append(comment.text.strip())

    return final_comments

if __name__ == '__main__':
    # 电影id
    movie_id = "35797709"
    # 电影评论数量
    num = 20
    comments = get_comments_by_id(movie_id, num)
    print(comments)
    print(len(comments))
    