import argparse
import os
import numpy
import cv2
import imutils
from imutils import paths
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import util
import csv

def stitch4(filename):
    im1 = Image.open(filename[0])
    #print("shape", np.shape(im1))
    im2 = Image.open(filename[1])
    im3 = Image.open(filename[2])
    im4 = Image.open(filename[3])
    dst = Image.new('RGBX', (im1.width * 2, im1.height * 2))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.width))
    dst.paste(im3, (im1.height, 0))
    dst.paste(im4, (im1.height, im1.width))
    #print("dst shape", np.shape(dst))
    return dst

def blendHaze(filename, dst):
    print("filename", filename)
    #Image.open(filename).save('temp.jpg')
    im = Image.open(filename)
    #print("shape", np.shape(im))
    im1 = im.resize((im.width*2, im.height*2), Image.NEAREST)
    #print("shape", np.shape(im))
    #print("shape", np.shape(im1))
    #print("shape", np.shape(dst))
    result = Image.blend(dst, im1, 0.5)
    return result

def writeNewCsv(label_dict, csv_file):
    csv_columns = ['image_names','tags']
    #print("label_dict", label_dict)
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_names', 'tags'])
        for key, value in label_dict.items():
            writer.writerow([key, value])
    csvfile.close()
    return


def main(filedir):
    image_path = r'C:\mypythonfiles\train-jpg'
    csv_path = r'C:\mypythonfiles\cs299_project\train_v2.csv'
    save_path = r'C:\mypythonfiles\cs299_project\stitched'
    labels_dict = util.load_labels(csv_path)
    keys = list(labels_dict.keys())
    length_dict = len(labels_dict)
    output_dict = {}
    j = 0
    filedir = image_path
    for i in range(0,int(length_dict / 4)):
        filenames = keys[i*4: i*4 + 4]
        filenameCom = []
        filenameCom.append(os.path.join(filedir, filenames[0] +'.jpg'))
        filenameCom.append(os.path.join(filedir, filenames[1] +'.jpg'))
        filenameCom.append(os.path.join(filedir, filenames[2] +'.jpg'))
        filenameCom.append(os.path.join(filedir, filenames[3] +'.jpg'))

        #print('filenames', filenames)
        stitched = stitch4(filenameCom)
        name_of_stitched = save_path+r'\stitch_'+str(i)+'.jpg'

        count = np.zeros(3)
        count[1] = 0.02 # human intervention has higher risk
        count[2] = 0.01
        new_label = ''
        for k in range(4):
            label = labels_dict[filenames[k]]
            if ('cloudy' in label and 'partly' not in label) or 'haze' in label:
                count[0] += 1
                selected_im = filenameCom[k]
            elif 'habitation' in label or 'agriculture' in label or 'cultivation' in label or 'conventional_mine' in label or 'selective_logging' in label or 'artisinal_mine' in label or 'slash_burn' in label:
                count[1] += 1
                new_label += label
            else:
                count[2] += 1
                new_label += label
            new_label += ' '
        #print("new_label", new_label)
        #print("count", count)
        max_count = np.max(count)
        #print("max count index", np.max(count))
        # BROKEN: blendHaze as dst as 3 channels and input has 4 channels
        if count[0] == max_count:
            blendHaze(selected_im, stitched).save(name_of_stitched)
            new_label = 'cloud haze'
        else:
            if count[1] == 0:
                new_label = 'primary'
            stitched.save(name_of_stitched)
        #print("label", label)
        output_dict['stitch_'+str(i)] = new_label
        j += 1
        if i > 2000 - 1:
            break
    writeNewCsv(output_dict, 'stitched_labels.csv')
    print("Wrote images:", i)
    util.load_labels('stitched_labels.csv')
    return

if __name__ == '__main__':
    main('Images')
