import os
from shutil import copyfile

# Copies a subset of images from source directory to dest directory
source_folder = "/home/charles/Downloads/train-jpg"
dest_folder = "/home/charles/me/academia/scpd/cs229/proj/train_subset"
image_prefix = "/train_"
image_suffix = ".jpg"
files_to_copy = 500
file_index_to_start = 1

def main():
    for i in range(files_to_copy):
        copyfile(source_folder + image_prefix + str(i + file_index_to_start) + image_suffix,
                 dest_folder + image_prefix + str(i + file_index_to_start) + image_suffix)
    return


if __name__ == "__main__":
    main()

