import re
import numpy as np
from random import shuffle
import cv2
import shutil
import os
import pymongo                       
from bson import ObjectId
import base64
from PIL import Image
from tqdm import tqdm


client = pymongo.MongoClient("mongodb://localhost:27017") 
DB = client["db_mongoocr"]
#print(DB.list_collection_names())
coll_img = DB.img_data
coll_info = DB.img_info
#dati_camera_b = coll_info.find({"metadata.CameraName": "B" })
def taglia_salva_test_img():
    path_test_img = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/test/"
    nomi_test_file = os.listdir(path_test_img+"annots/")
    for file in nomi_test_file:
        filename = file.split(".")[0] +".jpg"
        print(filename)
        dati_img = coll_info.find_one({"filename": filename })
        id_file = dati_img["_id"]
        rec_img = coll_img.find({"files_id": ObjectId(id_file)})
        binary_data = rec_img[0]["data"]
        dato_img_b64 = base64.b64encode(binary_data).decode('utf-8')
        print("RITAGLIO")
        x,y,w,h = 480, 170, 1900, 1410
        percorso_output = "D:\\Lavori_Unimore\\Dati_Marcegaglia\\immagini_estratte\\esempi_cameraB\\img_tagliate_rit_span\\"
        im_bytes = base64.b64decode(dato_img_b64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        x,y,w,h = 480, 170, 1900, 1410
        crop_image = img[y:h,x:w]
        cv2.imwrite(path_test_img + "/image/" + filename, crop_image)
        cv2.imwrite(percorso_output + filename, crop_image)




def sposta_img():
    path_train_img = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/train/"
    path_val_img = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/val/"
    path_img_to_copy = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/test/image_all/"
    path_annots_to_copy = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/test/annots_all/"
    path_test_img = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/test/image/"
    path_test_annots = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/test/annots/"

    nomi_test_file = os.listdir(path_annots_to_copy)
    shuffle(nomi_test_file)
    for file in nomi_test_file[0:1200]:
        with Image.open(path_img_to_copy+file.split(".")[0] + '.jpg') as img:
            output_filename = path_test_img + file.split(".")[0] + '.png'
            img.save(output_filename, format='PNG')

        annots = path_annots_to_copy + file
        shutil.copy(annots,path_test_annots)
"""
    nomi_train_file = os.listdir(path_train_img+"annots/")
    nomi_val_file = os.listdir(path_val_img+"annots/")
    for file in nomi_val_file:
        img = path_img_to_copy + file.split(".")[0] + ".png"
        shutil.copy(img, path_val_img+"image/")

    for file in nomi_train_file:
        img = path_img_to_copy + file.split(".")[0] + ".png"
        shutil.copy(img, path_train_img + "image/")
"""

def parse_test_file(file_path):
    data = []
    all_images = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            print(line)
            line = line.strip().replace(" ", "")
            parts = line.strip().split(':')
            if len(parts) == 2:
                image_name = parts[0]
                all_images.append(image_name)
                num_and_coordinates = parts[1].split('|')

                if len(num_and_coordinates) == 2:
                    num = int(num_and_coordinates[0].split('-->')[0])
                    den = int(num_and_coordinates[1].split('-->')[0])
                    coordinates_A = [tuple(map(int, point.strip('()').split(','))) for point in num_and_coordinates[0].split('-->')[1].split(';')]
                    coordinates_B = [tuple(map(int, point.strip('()').split(','))) for point in num_and_coordinates[1].split('-->')[1].strip('.').split(';')]

                    data.append({
                        'image_name': image_name,
                        'num': num,
                        'den': den,
                        'coordinates_A': coordinates_A,
                        'coordinates_B': coordinates_B
                    })

    file = open('marc_test.txt','w')
    for item in all_images:
        file.write("*"+item+"\n")
    file.close()

    return data


def parse_file(file_path):
    data = []
    all_images = []

    with open(file_path, 'r') as file:
        for line in file:
            print(line)
            line = line.strip().replace(" ", "")
            parts = line.strip().split(':')
            if len(parts) == 2:
                image_name = parts[0]
                all_images.append(image_name)
                num_and_coordinates = parts[1].split('|')

                if len(num_and_coordinates) == 2:
                    num = int(num_and_coordinates[0].split('-->')[0])
                    den = int(num_and_coordinates[1].split('-->')[0])
                    coordinates_A = [tuple(map(int, point.strip('()').split(','))) for point in num_and_coordinates[0].split('-->')[1].split(';')]
                    coordinates_B = [tuple(map(int, point.strip('()').split(','))) for point in num_and_coordinates[1].split('-->')[1].strip('.').split(';')]

                    data.append({
                        'image_name': image_name,
                        'num': num,
                        'den': den,
                        'coordinates_A': coordinates_A,
                        'coordinates_B': coordinates_B
                    })

    # train/test list
    shuffle(all_images)
    val_images = all_images[:1000]
    train_images = all_images[1000:]

    file = open('marc_train2.txt','w')
    for item in train_images:

        file.write("*"+item+"\n")
    file.close()

    file = open('marc_validation2.txt','w')
    for item in val_images:
        file.write("*"+item+"\n")
    file.close()

    return data

def to_yolo_sc(annots):
    # <object-class> <x_center> <y_center> <width> <height>
    w = 1420
    h = 1240

    for annot in annots:
        n_bb = np.array(annot['coordinates_A'])
        d_bb = np.array(annot['coordinates_B'])
        bb_all = np.concatenate([n_bb, d_bb], 0)

        xmin, ymin = bb_all.min(0)
        xmax, ymax = bb_all.max(0)

        # convert to yolo
        bw = (xmax-xmin)
        bh = (ymax-ymin)
        cx = xmin + bw/2
        cy = ymin + bh/2

        # normalize
        cx, cy = cx / w, cy / h
        bw, bh = bw / w, bh / h

        # save to file
        dest_file =  'dataset/'+annot['image_name'].split('.')[0] + '.txt'
        with open(dest_file, 'w') as f:
            out_str = f"0 {cx} {cy} {bw} {bh}"
            f.write(out_str)

def test_to_dota_not(annots):
    w = 1420
    h = 1240
    for annot in annots:
        nx1 = annot["coordinates_A"][0][0]
        ny1 = annot["coordinates_A"][0][1]
        nx2 = annot["coordinates_A"][1][0]
        ny2 = annot["coordinates_A"][1][1]
        nx3 = annot["coordinates_A"][2][0]
        ny3 = annot["coordinates_A"][2][1]
        nx4 = annot["coordinates_A"][3][0]
        ny4 = annot["coordinates_A"][3][1]

        dx1 = annot["coordinates_B"][0][0]
        dy1 = annot["coordinates_B"][0][1]
        dx2 = annot["coordinates_B"][1][0]
        dy2 = annot["coordinates_B"][1][1]
        dx3 = annot["coordinates_B"][2][0]
        dy3 = annot["coordinates_B"][2][1]
        dx4 = annot["coordinates_B"][3][0]
        dy4 = annot["coordinates_B"][3][1]

        dest_file = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/test/annots/"+ annot['image_name'].split('.')[0] + '.txt'
        with open(dest_file, 'w+') as f:
                # save to file
                out_str = f"{nx1} {ny1} {nx2} {ny2} {nx3} {ny3} {nx4} {ny4} num 0\n{dx1} {dy1} {dx2} {dy2} {dx3} {dy3} {dx4} {dy4} den 0"
                f.write(out_str)


def to_dota_not(annots):
    w = 1420
    h = 1240
    shuffle(annots)
    annots_val = annots[:1000]
    annots_train = annots[1000:]
    print(len(annots_val))
    print(len(annots_train))

    for annot in annots_val:
        nx1 = annot["coordinates_A"][0][0]
        ny1 = annot["coordinates_A"][0][1]
        nx2 = annot["coordinates_A"][1][0]
        ny2 = annot["coordinates_A"][1][1]
        nx3 = annot["coordinates_A"][2][0]
        ny3 = annot["coordinates_A"][2][1]
        nx4 = annot["coordinates_A"][3][0]
        ny4 = annot["coordinates_A"][3][1]

        dx1 = annot["coordinates_B"][0][0]
        dy1 = annot["coordinates_B"][0][1]
        dx2 = annot["coordinates_B"][1][0]
        dy2 = annot["coordinates_B"][1][1]
        dx3 = annot["coordinates_B"][2][0]
        dy3 = annot["coordinates_B"][2][1]
        dx4 = annot["coordinates_B"][3][0]
        dy4 = annot["coordinates_B"][3][1]

        #dest_file = './dataset/' + annot['image_name'].split('.')[0] + '.txt'
        dest_file = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/val/annots/"+ annot['image_name'].split('.')[0] + '.txt'
        with open(dest_file, 'w+') as f:
                # save to file
                out_str = f"{nx1} {ny1} {nx2} {ny2} {nx3} {ny3} {nx4} {ny4} num 0\n{dx1} {dy1} {dx2} {dy2} {dx3} {dy3} {dx4} {dy4} den 0"
                f.write(out_str)

    for annot in annots_train:
        nx1 = annot["coordinates_A"][0][0]
        ny1 = annot["coordinates_A"][0][1]
        nx2 = annot["coordinates_A"][1][0]
        ny2 = annot["coordinates_A"][1][1]
        nx3 = annot["coordinates_A"][2][0]
        ny3 = annot["coordinates_A"][2][1]
        nx4 = annot["coordinates_A"][3][0]
        ny4 = annot["coordinates_A"][3][1]

        dx1 = annot["coordinates_B"][0][0]
        dy1 = annot["coordinates_B"][0][1]
        dx2 = annot["coordinates_B"][1][0]
        dy2 = annot["coordinates_B"][1][1]
        dx3 = annot["coordinates_B"][2][0]
        dy3 = annot["coordinates_B"][2][1]
        dx4 = annot["coordinates_B"][3][0]
        dy4 = annot["coordinates_B"][3][1]

        dest_file = "C:/Users/iapi9/OneDrive/Desktop/LAVORI/Lavori_Unimore/Marcegaglia/mmrotate/tools/data/marc/train/annots/"+ annot['image_name'].split('.')[0] + '.txt'
        with open(dest_file, 'w+') as f:
                # save to file
                out_str = f"{nx1} {ny1} {nx2} {ny2} {nx3} {ny3} {nx4} {ny4} num 0\n{dx1} {dy1} {dx2} {dy2} {dx3} {dy3} {dx4} {dy4} den 0"
                f.write(out_str)


if __name__ == '__main__':
    # Example usage:
    #file_path = 'D:\\Lavori_Unimore\\Dati_Marcegaglia\\immagini_estratte\\esempi_cameraB\\coordinate_dati_ok.txt'
    #parsed_data = parse_test_file(file_path)
    #to_yolo_mc(parsed_data)
    #test_to_dota_not(parsed_data)
    #taglia_salva_test_img()
    sposta_img()