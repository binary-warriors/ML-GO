from mlgo import create_app
import os

proxy = 'http://edcguest:edcguest@172.31.100.14:3128'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
