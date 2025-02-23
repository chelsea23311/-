import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import os.path
import glob
import random
import shutil
folders=[  r'C:\Users\15342\Desktop\cats\CAT_00',
           r'C:\Users\15342\Desktop\cats\CAT_01',
           r'C:\Users\15342\Desktop\cats\CAT_02',
           r'C:\Users\15342\Desktop\cats\CAT_03',
           r'C:\Users\15342\Desktop\cats\CAT_04',
           r'C:\Users\15342\Desktop\cats\CAT_05']
def convertjpg(jpgfile, outdir, width=500, height=500):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width, height))
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for folder in folders:
    for jpgfile in glob.glob((folder + '/*.jpg')):
        convertjpg(jpgfile,r"C:\zeqi\three\new_cats")
def rate(file_name,folder):
    infile=os.path.join(folder, file_name)
    im=Image.open(infile)
    (x,y)=im.size
    a,b=(x,y)
    return x*y/(500*500)
for folder in folders:
    for txtfile in glob.glob((folder + '/*.txt')):
        file_name=os.path.splitext(txtfile)[0]
        r=rate(file_name,folder)
        f=open(txtfile,"r",encoding="UTF-8")
        out_path=os.path.join(r"C:\zeqi\three\new_cats",os.path.basename(txtfile))
        f1=open(out_path,"w",encoding="UTF-8")
        for line in f.readlines():
            numbers = line.strip().split()
            result_numbers = []
            for index, number in enumerate(numbers):
                if index ==0:
                    result_numbers.append(number)
                else:
                    number_float=float(number)
                    new_number=number_float * r
                    result_numbers.append(f"{new_number:.3f}")
            f1.write(" ".join(result_numbers)+"\n")
        f.close()
        f1.close()
txtfiles = glob.glob((r'C:\zeqi\three\new_cats' + '/*.txt'))
random.shuffle(txtfiles)
count=int(len(txtfiles)*0.8)
train_files=txtfiles[:count]
val_files=txtfiles[count:]
for txtfile in train_files:
    source_path=os.path.join(r"C:\zeqi\three\new_cats",os.path.basename(txtfile))
    target_path=os.path.join(r"C:\zeqi\three\train",os.path.basename(txtfile))
    shutil.move(source_path,target_path)
    file_name = os.path.splitext(txtfile)[0]
    jpg_file_path = glob.glob(os.path.join(r"C:\zeqi\three\new_cats", file_name))
    if jpg_file_path:
        source_path_jpg = jpg_file_path[0]
        target_path_jpg = os.path.join(r"C:\zeqi\three\train", os.path.basename(source_path_jpg))
        shutil.move(source_path_jpg, target_path_jpg)
for txtfile in glob.glob((r'C:\zeqi\three\new_cats'+ '/*.txt')):
    source_path=source_path=os.path.join(r"C:\zeqi\three\new_cats",os.path.basename(txtfile))
    target_path = os.path.join(r"C:\zeqi\three\val", os.path.basename(txtfile))
    shutil.move(source_path, target_path)
    file_name = os.path.splitext(txtfile)[0]
    jpg_file_path = glob.glob(os.path.join(r"C:\zeqi\three\new_cats", file_name))
    if jpg_file_path:
        source_path_jpg = jpg_file_path[0]
        target_path_jpg = os.path.join(r"C:\zeqi\three\val", os.path.basename(source_path_jpg))
        shutil.move(source_path_jpg, target_path_jpg)
for jpgfile in glob.glob((r'C:\Users\15342\Desktop\cats_1\cats\test'+ '/*.jpg')):
    convertjpg(jpgfile, r"C:\zeqi\three\test",500,500)