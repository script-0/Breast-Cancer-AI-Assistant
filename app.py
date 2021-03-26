from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our Breast Cancer Assistant API !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5002)
