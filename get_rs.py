
import multiprocessing as mp
import requests
from urllib.parse import quote
import urllib3
import time
import os
import json


def write_txt(dst_path, dst_list):
    with open(dst_path, 'w') as txt:
        for line in dst_list:
            #print('***', line)
            txt.write(line+'\n')


def listdir(path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            else:
                list_name.append(file_path)

def get_det_rs(pic_path):

    #pic_path = quote(pic_path)
    url = 'http://0.0.0.0:8000/det?url={}'.format(pic_path)
    r = requests.get(url)
    rs_status = r.status_code
    info = 'null'
    if rs_status==200:
        rs = r.json()
        info = json.dumps(rs)
    return info
    
    #http = urllib3.PoolManager()
    #r = http.request('GET', url)
    #r_status = r.status
    #print(r.data)
    #print(r_status)

def get_pic_list(fold):
    files = os.walk(fold)
    return files


if __name__ == '__main__':
    pic_fold = '/home/tang/linux/data/coco2017/val2017'
    dst_path = '/home/tang/linux/data/predict/coco_val_pred.txt'
    pic_list = []
    listdir(pic_fold, pic_list)
    #pic_list = pic_list[:1000]

    print('pic nums:', len(pic_list))
    time_s = time.time()
    pool = mp.Pool(processes=12)

    rs_list = []
    for i, pic_path in enumerate(pic_list):
        rs_list.append(pool.apply_async(get_det_rs, (pic_list[i],) ))
        #rs_list.append(get_det_rs(pic_path))
    pool.close()
    pool.join()
    time_e = time.time()
    
    str_rs_list = [r.get() for r in rs_list if r.get()!='null']
    write_txt(dst_path, str_rs_list)

    print('Pic Num=',len(pic_list) , 'pics | Time=', round((time_e-time_s)/60, 1), 'mims | QPS=', round(len(pic_list)/(time_e-time_s), 1), 'img/s' )
    
    
        
        
    
    

