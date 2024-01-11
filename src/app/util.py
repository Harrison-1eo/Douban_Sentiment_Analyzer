import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
import wordcloud
import sys
sys.path.append("..")
sys.path.append("src")
from preprocessing import preprocessing as pp

# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['simsun', 'Kaitt', 'SimHei']
 
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False


def draw_pie_chart(data, labels, title, save_path):
    # 设置字体
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")

    # 设置图形大小
    plt.figure(figsize=(5, 5), dpi=100)

    # 绘制饼状图
    plt.pie(data, labels=labels, autopct="%1.2f%%", colors=["red", "blue", "green"])

    # 添加图例
    plt.legend(prop=my_font)

    # 保存图片
    plt.savefig(save_path)

    # 展示图片
    # plt.show()
    plt.close()


def draw_cloud(data, save_path):
    # data: list of str
    # 1. 分词
    data_cut = pp.cut_words(data)
    data_cut = pp.remove_stop_words(data_cut, 'res\dicts\stop-words.txt')
    cut_text = " ".join([" ".join(c) for c in data_cut])
    
    # 2. 生成词云
    wc = wordcloud.WordCloud(
        font_path="C:\Windows\Fonts\simsun.ttc",
        background_color="white",
        width=1000,
        height=700,
        max_words=100,
        max_font_size=100,
        min_font_size=10
    )
    wc.generate(cut_text)

    # 3. 展示词云
    plt.imshow(wc)

    # 4. 保存词云
    plt.savefig(save_path)

    # 5. 展示词云
    # plt.show()
    plt.close()


if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv("comments_res.txt", encoding="utf-8")

    # 绘制饼状图
    draw_pie_chart(
        data=[len(df[df["scores"] > 0]), len(df[df["scores"] < 0]), len(df[df["scores"] == 0])],
        labels=["正面评论", "负面评论", "中性评论"],
        title="豆瓣电影《我和我的家乡》短评情感分析结果",
        save_path="pie_chart.png"
    )

    # 绘制词云
    draw_cloud(df["comments"].tolist(), "cloud.png")