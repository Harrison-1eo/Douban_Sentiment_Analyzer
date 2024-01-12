import sys
import os

sys.path.append(".")
sys.path.append("src")

import tkinter.messagebox as messagebox
try:
    import tkinter as tk
    from crawling import douban_comments_get as dcg
    from crawling import douban_info_get as dig
    from emotion_analysis.SO_PMI import SO_PMI_cal as spc
    from emotion_analysis.lexicon_weighted import LWS as lws
    from util import *
    from PIL import Image,ImageTk
    import pandas, numpy, matplotlib, sklearn, jieba, gensim, wordcloud
    from bs4 import BeautifulSoup
except Exception as e:
    print(e)
    print("请安装所需的依赖库")
    messagebox.showinfo("提示", "请安装所需的依赖库\n"+str(e))
    exit()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("My App")
        self.geometry("400x300")
        self.resizable(0, 0)

        # 添加一个标签：欢迎
        self.label = tk.Label(self, text="豆瓣电影短评情感分析系统", font=("微软雅黑", 20))
        self.label.pack(pady=10)

        # 添加一个按钮：豆瓣电影短评分析
        self.btn_douban = tk.Button(self, text="豆瓣电影短评分析", command=self.douban,
                                     width=15, height=3,bg="#5cb85c",
                                     fg="#FFFFFF", activebackground="#4cae4c", activeforeground="#FFFFFF",
                                     font=("微软雅黑", 12))
        self.btn_douban.pack(pady=10)

        # 添加一个按钮：关于
        self.btn_about = tk.Button(self, text="关于", command=self.about,
                                   width=15, height=3,bg="#5bc0de",
                                   fg="#FFFFFF", activebackground="#46b8da", activeforeground="#FFFFFF",
                                   font=("微软雅黑", 12))
        self.btn_about.pack(pady=10)

        # 检测关闭窗口事件
        def close():
            self.destroy()
            exit()
        self.protocol("WM_DELETE_WINDOW", close)

    def about(self):
        # 一个新窗口，用于显示关于信息
        about_window = tk.Toplevel(self)
        about_window.title("关于")
        about_window.geometry("300x200")
        about_window.resizable(0, 0)

        def on_closing():
            self.deiconify()
            about_window.destroy()

        about_window.protocol("WM_DELETE_WINDOW", on_closing)
        self.withdraw()

        # 关于信息
        about_text = tk.Label(about_window, text="这是一个关于信息")
        about_text.pack()

    def douban(self):
        # 一个新窗口，用于显示豆瓣电影短评分析
        douban_window = tk.Toplevel(self)
        douban_window.title("豆瓣电影短评分析")
        douban_window.geometry("300x350")
        douban_window.resizable(0, 0)
        douban_window.resizable(True, True)

        def on_closing():
            self.deiconify()
            douban_window.destroy()

        douban_window.protocol("WM_DELETE_WINDOW", on_closing)
        self.withdraw()

        # 一个输入框，用于输入电影编号
        text = tk.Label(douban_window, text="请输入电影编号：", font=("微软雅黑", 12))
        text.pack()
        douban_input = tk.Entry(douban_window)
        douban_input.pack(pady=10)

        douban_input.insert(0, "1292722")

        # 一个标签，用于显示对电影编号的解释
        explain_msg = "电影编号是电影在豆瓣网站上的唯一标识\n" \
                        "可以在电影的网址中找到\n" \
                        "例如：https://movie.douban.com/subject/1292722/\n" \
                        "电影编号为：1292722\n" \
                        "提供几个电影编号：1292722、1292052、1292789"
        text = tk.Label(douban_window, text=explain_msg, font=("微软雅黑", 8))
        text.pack()

        # 一个选项，用于选择聚类算法
        # 一个标签，用于显示聚类算法
        # 标签和选项在同一行显示
        cluster_frame = tk.Frame(douban_window)
        cluster_frame.pack(pady=10)

        text = tk.Label(cluster_frame, text="请选择聚类算法：")
        cluster_algorithm = tk.StringVar()
        cluster_algorithm.set("K-Means")
        cluster_algorithm_menu = tk.OptionMenu(cluster_frame, cluster_algorithm, "DBSCAN", "K-Means", "LDA")

        text.pack(side=tk.LEFT)
        cluster_algorithm_menu.pack(side=tk.LEFT)

        # 一个选项，用于选择情感分析算法
        # 一个标签，用于显示情感分析算法
        # 标签和选项在下一行显示
        emotion_frame = tk.Frame(douban_window)
        emotion_frame.pack(pady=10)

        text = tk.Label(emotion_frame, text="请选择情感分析算法：")
        # 一个下拉菜单，用于选择情感分析算法fv  qij-;/[x ]
        emotion_algorithm = tk.StringVar()
        emotion_algorithm.set("SO-PMI")
        emotion_algorithm_menu = tk.OptionMenu(emotion_frame, emotion_algorithm, "SO-PMI", "朴素字典", "TextCNN")
        
        text.pack(side=tk.LEFT)
        emotion_algorithm_menu.pack(side=tk.LEFT)


        # 一个按钮，用于开始分析，点击后调用 douban_analyze 函数
        douban_btn = tk.Button(douban_window, text="开始分析", 
                               command=lambda: self.douban_get_data(
                                   douban_input.get(), 
                                   {
                                        "cluster_algorithm": cluster_algorithm.get(),
                                        "sentiment_algorithm": emotion_algorithm.get()
                                    }
                                   ),
                                width=10, height=2, font=("微软雅黑", 12)
                                )
        douban_btn.pack(pady=10)


    def douban_get_data(self, movie_id, func_args):
        if os.path.exists("save/" + movie_id):
            # 弹窗提示，已经抓取过了
            messagebox.showinfo("提示", "已经抓取过了，将展示结果")
            with open("save/" + movie_id + "/comments_res.csv", "r", encoding="utf-8") as f:
                comments = pd.read_csv(f)
                res = self.douban_analyze(movie_id, comments["comments"].tolist(), func_args)
                self.douban_show_res(res)
            
        else:
            comments = dcg.get_comments_by_id(movie_id, 500)

            if comments is None:
                # 弹窗提示，抓取失败
                messagebox.showinfo("提示", "抓取失败")
                return
            elif comments == "IP":
                messagebox.showinfo("提示", "IP被封了，请在douban_comments_get.py中更换代理IP，或者使用油猴插件等方式")
            else:
                # 弹窗提示，抓取成功
                messagebox.showinfo("提示", "抓取成功")
                res = self.douban_analyze(movie_id, comments, func_args)
                # 展示分析结果
                self.douban_show_res(res)

    def douban_analyze(self, id, comments, func_args):
        scores = []
        if func_args["sentiment_algorithm"] == "SO-PMI":
            sopmi = spc.SoPmiSentiment()
            for index, c in enumerate(comments):
                c_cut = sopmi.word_cut(c)
                scores.append(sopmi.score(c_cut))
        elif func_args["sentiment_algorithm"] == "朴素字典":
            lws_ = lws.LexiconWeightedSentiment()
            scores = lws_.analyze_sentiment_score(comments)
        elif func_args["sentiment_algorithm"] == "TextCNN":
            from emotion_analysis.TextCNN import TextCNN_cal as tc
            scores = tc.predict(comments)
            scores = [1 if s > 0 else -1 for s in scores]
        else:
            messagebox.showinfo("提示", "未知的情感分析算法")
            return None

        good_comments = [c for index, c in enumerate(comments) if scores[index] > 0]
        bad_comments = [c for index, c in enumerate(comments) if scores[index] < 0]
        mid_comments = [c for index, c in enumerate(comments) if scores[index] == 0]

        movie_info = dig.get_info_by_id(str(id))

        # 返回分析结果
        res = {
            "id": id,
            "info": movie_info,
            "sentiment_algorithm": func_args["sentiment_algorithm"],
            "cluster_algorithm": func_args["cluster_algorithm"],
            "comments": comments,
            "scores": scores,
            "good_comments": good_comments,
            "bad_comments": bad_comments,
            "mid_comments": mid_comments
        }

        return res
    
    def douban_show_res(self, res):
        global pie_img
        global cloud_img
        global cluster_img
        # 保存到文件的路径，保存在当前目录下的 save 文件夹中，文件夹不存在则创建，
        # 为了防止文件名重复，文件名为当前时间戳，格式为：年-月-日-时-分-秒
        path = "save/" + res['id'] + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        comments_scores = [(res['comments'][index], res['scores'][index]) for index, c in enumerate(res['comments'])]
        # 保存到文件
        with open(path + "comments_res.csv", "w", encoding="utf-8") as f:
            f.write("comments,scores\n")
            for i in range(len(comments_scores)):
                f.write(f"{comments_scores[i][0]},{comments_scores[i][1]}\n")

        # 新建一个窗口，用于展示分析结果
        douban_res_window = tk.Toplevel(self)
        douban_res_window.title("分析结果")
        douban_res_window.geometry("1100x800")
        douban_res_window.resizable(0, 0)
        # 允许用户调节窗口大小
        douban_res_window.resizable(True, True)

        header_frame = tk.Frame(douban_res_window)
        header_frame.pack(pady=10, padx=20, anchor=tk.W)

        info_frame = tk.Frame(header_frame)
        info_frame.pack(pady=10, side=tk.LEFT)

        # 一个标签，用于显示电影名称
        name_text = tk.Label(info_frame, text=f"{res['info']['name']}", font=("微软雅黑", 20))
        director_text = tk.Label(info_frame, text=f"导演：{res['info']['director']}")
        scriptwriter_text = tk.Label(info_frame, text=f"编剧：{res['info']['scriptwriter']}")
        actors_text = tk.Label(info_frame, text=f"主演：{res['info']['actors']}")
        genre_text = tk.Label(info_frame, text=f"类型：{res['info']['genre']}")
        country_text = tk.Label(info_frame, text=f"制片国家/地区：{res['info']['country']}")
        language_text = tk.Label(info_frame, text=f"语言：{res['info']['language']}")
        release_date_text = tk.Label(info_frame, text=f"上映日期：{res['info']['release_date']}")

        # 每个标签一行显示，左对齐
        name_text.pack(anchor=tk.W)
        director_text.pack(anchor=tk.W)
        scriptwriter_text.pack(anchor=tk.W)
        actors_text.pack(anchor=tk.W)
        genre_text.pack(anchor=tk.W)
        country_text.pack(anchor=tk.W)
        language_text.pack(anchor=tk.W)
        release_date_text.pack(anchor=tk.W)

        num_frame = tk.Frame(header_frame, width=200)
        num_frame.pack(pady=10, padx=200, anchor=tk.CENTER, side=tk.RIGHT)

        # 一个标签，用于显示正面评论数
        good_comments_text = tk.Label(num_frame, text=f"正面评论数：{len(res['good_comments'])}")
        good_comments_text.pack()

        # 一个标签，用于显示负面评论数
        bad_comments_text = tk.Label(num_frame, text=f"负面评论数：{len(res['bad_comments'])}")
        bad_comments_text.pack()

        # 一个标签，用于显示中性评论数
        mid_comments_text = tk.Label(num_frame, text=f"中性评论数：{len(res['mid_comments'])}")
        mid_comments_text.pack()

        # 载入并缩放图像
        def load_and_resize_image(image_path, size=(200, 200)):
            original_image = Image.open(image_path)
            resized_image = original_image.resize(size, Image.Resampling.LANCZOS)
            return resized_image

        # 展示饼状图
        if res["sentiment_algorithm"] != "TextCNN":
            draw_pie_chart(
                [len(res['good_comments']), len(res['bad_comments']), len(res['mid_comments'])],
                ["正面评论", "负面评论", "中性评论"],
                "豆瓣电影短评情感分析结果",
                path + "pie.png"
            )
        else:
            draw_pie_chart(
                [len(res['good_comments']), len(res['bad_comments'])],
                ["正面评论", "负面评论"],
                "豆瓣电影短评情感分析结果",
                path + "pie.png"
            )
        
        # pic_frame 换行居中显示
        pic_frame = tk.Frame(douban_res_window)
        pic_frame.pack(pady=10, padx=20, anchor=tk.CENTER)

        pie_frame = tk.Frame(pic_frame)
        pie_frame.pack(side=tk.LEFT)

        cloud_frame = tk.Frame(pic_frame)
        cloud_frame.pack(side=tk.LEFT)

        cluster_frame = tk.Frame(pic_frame)
        cluster_frame.pack(side=tk.LEFT)

        # 一个标签，用于显示饼状图
        pie_text = tk.Label(pie_frame, text="情感分析饼状图")
        pie_text.pack()

        # 一个标签，用于显示饼状图
        pie_img = ImageTk.PhotoImage(load_and_resize_image(path + "pie.png", (250, 250)))
        pie_img_label = tk.Label(pie_frame, image=pie_img)
        pie_img_label.pack()
        
        # 展示词云图
        draw_cloud(res['comments'], path + "cloud.png")
        # 一个标签，用于显示词云图
        cloud_text = tk.Label(cloud_frame, text="词云图")
        cloud_text.pack()

        # 一个标签，用于显示词云图
        cloud_img = ImageTk.PhotoImage(load_and_resize_image(path + "cloud.png", (400, 250)))
        cloud_img_label = tk.Label(cloud_frame, image=cloud_img)
        cloud_img_label.pack()

        # 展示聚类图
        try:
            if res["cluster_algorithm"] == "K-Means":
                from cluster import kmeans as km
                km.draw_kmeans(res['comments'], path + "cluster.png", res['info']['name'])
            elif res["cluster_algorithm"] == "DBSCAN":
                from cluster import dbscan as db
                db.draw_dbscan(res['comments'], path + "cluster.png", res['info']['name'])
            elif res["cluster_algorithm"] == "LDA":
                from cluster import LDA as ld
                ld.draw_LDA(res['comments'], path + "cluster.png", res['info']['name'])
            else:
                messagebox.showinfo("提示", "未知的聚类算法")

                
            # 一个标签，用于显示聚类图
            cluster_text = tk.Label(cluster_frame, text="聚类图")
            cluster_text.pack()

            # 一个标签，用于显示聚类图
            cluster_img = ImageTk.PhotoImage(load_and_resize_image(path + "cluster.png", (400, 250)))
            cluster_img_label = tk.Label(cluster_frame, image=cluster_img)
            cluster_img_label.pack()
        except Exception as e:
            print(e)
            print("聚类失败")
            # messagebox.showinfo("提示", "聚类失败")

        best_comments_frame = tk.Frame(douban_res_window)
        best_comments_frame.pack(pady=10, padx=20, anchor=tk.W)

        # 显示正面评论得分最高的三条评论
        if res["sentiment_algorithm"] != "TextCNN":
            
            good_comments_text = tk.Label(best_comments_frame, text="正面评论得分最高的几条评论：")
            good_comments_text.pack(anchor=tk.W)
            comments_scores.sort(key=lambda x: x[1], reverse=True)
            for i in range(3):
                text=f"{comments_scores[i][0]} >>> 得分：{comments_scores[i][1]}"
                # 一个标签，用于显示正面评论得分最高的三条评论
                # 设置标签的宽度为1000，高度为自动调整，左对齐，自动换行
                good_comments_text = tk.Label(best_comments_frame, text=text, wraplength=1000, anchor=tk.W, justify=tk.LEFT)
                good_comments_text.pack(pady=10)

    
    def run(self):
        self.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    app = App()
    app.run()