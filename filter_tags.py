import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import util

def main():
    label_path = r'/home/anishag/tree/cs229/cs229TreeHugger/train_v2.csv'
    labels = util.load_labels(label_path)
    for key in labels.keys():
        print("labels: ", labels[key])
        label = labels[key]
        if 'cloudy' in label and 'partly_cloudy' not in label:
            print(label)
        elif 'habitation' in label or 'agriculture' in label or 'cultivation' in label or 'conventional_mine' in label or 'selective_logging' in label or 'artisinal_mine' in label or 'slash_burn' in label:
            print('human_intrusion')
        else:
            print('no_human_intrusion') 
    return

if __name__ == '__main__':
	main()
