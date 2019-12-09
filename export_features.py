import util
import featureExtract
import csv


img_folder = "train_subset"
img_folder = "/home/charles/Downloads/train-jpg"
csv_file = "train_v2.csv"
output_csv_file = "features.csv"


def main(train_path, label_path):
    # images, label_dict = util.load_data_and_label(train_path, label_path)
    image_to_feature_dict = util.extract_all_feature(train_path, featureExtract.extract_features)
    features_to_csv = []


    i = 0
    for img_name in image_to_feature_dict.keys():
        features = [img_name]
        features += image_to_feature_dict[img_name]

        features_to_csv += [features]
        if i % 1000 == 0:
            print("Working on iteration: " + str(i))
        i += 1

    with open(output_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(features_to_csv)


main(img_folder, csv_file)

