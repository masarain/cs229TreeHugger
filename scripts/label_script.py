import numpy as np


def load_labels(csv_path):
    with open(csv_path, 'r') as csv_file:
        headers = csv_file.readline().strip().split(',')

    result = []
    with open(csv_path, 'r') as csv_file:
        line = csv_file.readline().strip().split(',')
        line = csv_file.readline().strip().split(',')
        while line and len(line) == 2:
            result += [line[1]]
            line = csv_file.readline().strip().split(',')

    return result


def find_unique_catgories(all_categories):
    categories = []
    for tag in all_categories:
        category_list = tag.split(' ')
        for category in category_list:
            if not (category in categories):
                categories += [category]
    return categories


all_tags = load_labels("train_v2.csv")
unique_categories = find_unique_catgories(all_tags)
print(unique_categories)
