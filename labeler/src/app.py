

import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask.ext.cors import CORS
from pymongo import MongoClient
import json
import logging

app = Flask(__name__, template_folder='templates')
cors = CORS(app)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

client = MongoClient("mongodb://mongo-191809ff-1.b3d823aa.cont.dockerapp.io:32770")
db = client.epopcon
products = db.products


def get_products_lst(n=20):
    retrived_lst = list(products.find({'STATUS': 'waiting'}, {'_id': False}).limit(n))
    return retrived_lst

def get_products_by_status(status, n=100):
    retrived_lst = list(products.find({'STATUS': status}, {'_id': False}).limit(n))
    return retrived_lst

def update_status(status, good_nos, username):
    result = products.update_many({'GOODS_NO': {'$in': good_nos}}, {'$set': {'STATUS': status, 'USERNAME': username}})
    return result.acknowledged

@app.route('/')
def index():

    return render_template('gallery.html')

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)

        file.save(destination)
    return send_from_directory("images", filename, as_attachment=True)

@app.route('/save', methods=['POST'])
def save():


    data = json.loads(request.data)
    logging.warning(data['bundle'])
    logging.warning(data['username'])

    positive_list = []
    negative_list = []

    for item in data['bundle']:

        if item['STATUS'] == "1":
            positive_list.append(item['GOODS_NO'])
        else:
            negative_list.append(item['GOODS_NO'])


    update_status("1", positive_list, data['username'])
    update_status("-1", negative_list, data['username'])

    return ('', 200)

@app.route('/correct')
def correct():
    items = get_products_by_status('1')
    return render_template("correct.html", items=items)

@app.route('/incorrect')
def incorrect():
    items = get_products_by_status('-1')
    return render_template("incorrect.html", items=items)


@app.route('/update')
def update():
    products.update_many({'STATUS': 'progress'}, {'$set': {'STATUS': 'waiting'}})
    return ('', 200)

@app.route('/angulartest')
def angular():
    return render_template('angulartest.html')

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/items', methods=['GET'])
def get_items():



    image_names = get_products_lst()
    GOODS_NOs = [item['GOODS_NO'] for item in image_names]
    update_status('progress', GOODS_NOs, "")

    response = jsonify(image_names)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":

    app.run(host='0.0.0.0', port=80, debug=True)
