from flask import Flask, request

app = Flask(__name__)

@app.route('/receive_comments', methods=['POST'])
def receive_comments():
    data = request.form['data']
    # 处理数据
    # 存储到本地文件
    with open('server_comments.txt', 'a', encoding='utf-8') as f:
        f.write(data)

    return 'Comments received'

if __name__ == '__main__':
    app.run(port=5000)
