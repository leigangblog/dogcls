# -*- coding: utf-8 -*-
import base64
import requests
import json


def get_access_token():
    # 获取token的API
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    # 获取access_token需要的参数
    params = {
        # 固定参数
        'grant_type': 'client_credentials',
        # 必选参数，传入你的API Key
        'client_id': '你的API Key',
        # 必选参数，传入你的Secret Key
        'client_secret': '你的Secret Key'
    }
    # 发送请求，获取响应数据
    response = requests.post(url, params)
    # 将响应的数据转成字典类型，然后取出access_token
    access_token = response.json()['access_token']
    # 将access_token返回
    return access_token


def dogClsv1(img):
    '''
    easydl图像分类
    '''
    access_token = str(get_access_token())
    # 记得替换为你的url
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/dogClsv1' + "?access_token=" + access_token   
    # print(url)
    with open(img, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        img = base64_data.decode('UTF8')
    # 请求的headers信息，固定写法
    # headers = {'content-type': 'application/x-www-form-urlencoded'}
    headers = {'content-type': 'application/json'}
    # 请求的参数
    params = {
        "image": img,
        "top_num": "5"
    }
    params = json.dumps(params)
    # 发送请求
    response = requests.post(url, data=params, headers=headers)
    # print(response)
    res=[]
    # 对响应结果进行处理
    if response:
        res = response.json()
        name = res['results'][0]['name']
        score = res['results'][0]['score']
        return res,name,score



if __name__ == '__main__':
    # res={'log_id': 366203712637603713, 'results': [{'name': 'affenpinscher', 'score': 0.9862339496612549},
    #                                            {'name': 'Brabancon_griffo', 'score': 0.00653552683070302},
    #                                            {'name': 'Shih_Tzu', 'score': 0.0007479606429114938},
    #                                            {'name': 'Pekinese', 'score': 0.0005000945529900491},
    #                                            {'name': 'Bouvier_des_Flandres', 'score': 0.00032892366289161146}]}
    res,name,score=dogClsv1('./1/n107845.jpg')
    print("这是一只{0}狗，可信度为{1}.".format(name,score))
