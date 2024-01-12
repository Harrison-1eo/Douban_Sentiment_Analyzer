
import requests
from bs4 import BeautifulSoup

def get_info_by_id(id):

    url = f"https://movie.douban.com/subject/{id}/"
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


    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 定义一个函数来提取信息
    def extract_info(label):
        element = soup.find('span', class_='pl', text=label)
        if element and element.next_sibling:
            print(element, element.next_sibling)
            return element.next_sibling.strip(': ')
        return None
    
    # 定义一个函数来提取包含多个条目的信息
    def extract_multiple_info(label):
        element = soup.find('span', class_='pl', text=label)
        if element:
            return [a.get_text() for a in element.find_next('span', class_='attrs').find_all('a')]
        return None
    
    def extract_enhence_info(label):
        for l in soup.find_all('span', class_='pl'):
            if l.get_text().strip() == label:
                next_span = l.find_next_sibling('span')
                if next_span:
                    return next_span.get_text().strip()
        return None

    movie_name = soup.find('span', {'property': 'v:itemreviewed'}).get_text().strip()

    director = extract_multiple_info('导演')
    director = ','.join(director) if director else ''
    scriptwriter = extract_multiple_info('编剧')
    # 如果超过3个，显示前3个和省略号
    if scriptwriter and len(scriptwriter) > 3:
        scriptwriter = ','.join(scriptwriter[:3]) + '...'
    else:
        scriptwriter = ','.join(scriptwriter) if scriptwriter else ''
    actors = extract_multiple_info('主演')
    if actors and len(actors) > 3:
        actors = ','.join(actors[:3]) + '...'
    else:
        actors = ','.join(actors) if actors else ''
    genre = extract_enhence_info('类型:')
    country = extract_info('制片国家/地区:')
    language = extract_info('语言:')
    release_date = extract_enhence_info('上映日期:')
    duration = extract_enhence_info('片长:')
    alternate_name = extract_info('又名:')
    imdb = extract_info('IMDb:')

    # 打印提取的信息
    print(f"电影名称: {movie_name}")
    print(f"导演: {director}")
    print(f"编剧: {scriptwriter}")
    print(f"主演: {actors}")
    print(f"类型: {genre}")
    print(f"制片国家/地区: {country}")
    print(f"语言: {language}")
    print(f"上映日期: {release_date}")
    print(f"片长: {duration}")
    print(f"别名: {alternate_name}")
    print(f"IMDb编号: {imdb}")

    return {
        'name': movie_name,
        'director': director,
        'scriptwriter': scriptwriter,
        'actors': actors,
        'genre': genre,
        'country': country,
        'language': language,
        'release_date': release_date,
        'duration': duration,
        'alternate_name': alternate_name,
        'imdb': imdb,
    }

if __name__ == '__main__':
    # 电影id
    movie_id = "30270746"
    get_info_by_id(movie_id)