from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, send_file, make_response
from io import BytesIO
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
# from models.unet3plus import UNet_3Plus
import os
import cv2
from flask_cors import CORS

# device = 'cuda:3'
# model = UNet_3Plus(in_channels=1).to(device)
# model_weights = torch.load("./ckpt/unet_demo_7/best.pth")["model_state_dict"]
# model.load_state_dict(model_weights)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json;charset=utf-8'
CORS(app)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# def preprocess(image):
#     image = image / 255.0
#     image = np.array(cv2.resize(image, (256, 256)))[:, :, None].astype(np.float32)
#     image_tensor = transform(image).to(device)[None, :, :, :]
#     return image_tensor

# def postprocess(image, fn):
#     image = image[0, 0, :, :]
#     fig = plt.figure(figsize=(8,8))
#     plt.tight_layout()
#     plt.axis('off')
#     plt.xticks([]) 
#     plt.yticks([])
#     plt.imshow(image)
#     plt.savefig(os.path.join('./masks', fn), bbox_inches='tight', pad_inches=-0.1)


# @app.route('/upload', methods=['POST'])
# def upload():
#     files = request.files.getlist('files')
#     filenames = []
#     print(files)
#     for file in files:
#         print(file.filename)
#         file.save(os.path.join('uploads', file.filename))
#         image_data = cv2.imread(os.path.join('uploads', file.filename), cv2.IMREAD_GRAYSCALE)
#         print(type(image_data), image_data.shape)
#         image_tensor = preprocess(image_data)
#         output_tensor = model(image_tensor)
#         postprocess(output_tensor.detach().cpu().numpy(), file.filename)
#         filenames.append(file.filename)
#     json = jsonify({'filenames': filenames})
#     print('returns')
#     return json

def seg_preprocess(image_data):
    return image_data

def seg_postprocess(image_tensor):
    pass

@app.route('/segupload', methods=['POST'])
def segupload():
    files = request.files.getlist('files')
    filenames = []
    print(files)
    for file in files:
        print(file.filename)
        file.save(os.path.join('uploads', file.filename))
        image_data = cv2.imread(os.path.join('uploads', file.filename), cv2.IMREAD_GRAYSCALE)
        print(type(image_data), image_data.shape)
        image_tensor = seg_preprocess(image_data)
        output_tensor = seg_model(image_tensor)
        seg_postprocess(output_tensor.detach().cpu().numpy(), file.filename)
        filenames.append(file.filename)
    json = jsonify({'filenames': filenames})
    print('returns')
    return json

@app.route('/download/<string:filename>', methods=['GET', 'POST'])
def download(filename):
    if ',' in filename:
        filenames = [name.split('/')[-1] for name in filename.split(',')]
        image_datas = [open(os.path.join('uploads', '%s' % filename), "rb").read() for filename in filenames]
        response = make_response(image_data)
        response.headers['content-type'] = 'image/png'
        return response
    image_data = open(os.path.join('uploads', '%s' % filename), "rb").read()
    response = make_response(image_data)
    response.headers['content-type'] = 'image/png'
    return response

@app.route('/mask/<string:filename>', methods=['GET', 'POST'])
def mask(filename):
    if ',' in filename:
        filenames = [name.split('/')[-1] for name in filename.split(',')]
        image_datas = [open(os.path.join('masks', '%s' % filename), "rb").read() for filename in filenames]
        response = make_response(image_data)
        response.headers['content-type'] = 'image/png'
        return response
    image_data = open(os.path.join('masks', '%s' % filename), "rb").read()
    response = make_response(image_data)
    response.headers['content-type'] = 'image/png'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)

@app.route('/downloadZip')
def download_file():
    # 获取文件路径参数
    file_path = request.args.get('file_path')
    # 检查文件是否存在
    if os.path.isfile(file_path):
        # 返回文件对象
        return send_file(file_path, as_attachment=True)
    else:
        # 文件不存在，返回错误信息
        return 'File not found.', 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
