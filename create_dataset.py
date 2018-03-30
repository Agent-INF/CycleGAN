"""Create datasets for training and testing."""
import os
import argparse
import csv
import random
import utils


def create_list(foldername, fulldir=True, suffix=".jpg"):
  """
  :param foldername: The full path of the folder.
  :param fulldir: Whether to return the full path or not.
  :param suffix: Filter by suffix.
  :return: The list of filenames in the folder with given suffix.
  """
  file_list_tmp = os.listdir(foldername)
  file_list = []
  if fulldir:
    for item in file_list_tmp:
      if item.endswith(suffix):
        file_list.append(os.path.join(foldername, item))
  else:
    for item in file_list_tmp:
      if item.endswith(suffix):
        file_list.append(item)
  return file_list


def create_dataset(args):
  """
  saving data paths to csv file
  """
  image_dir_a = os.path.join('./data', args.dataset_name, args.mode + 'A')
  image_dir_b = os.path.join('./data', args.dataset_name, args.mode + 'B')
  list_a = create_list(image_dir_a, True, utils.IMG_TYPE)
  list_b = create_list(image_dir_b, True, utils.IMG_TYPE)

  output_path = os.path.join('./data', args.dataset_name,
                             '%s_%s.csv' % (args.dataset_name, args.mode))

  num_rows = max(len(list_a), len(list_b))
  all_data_tuples = []
  for i in range(num_rows):
    all_data_tuples.append((
        list_a[i % len(list_a)],
        list_b[i % len(list_b)]
    ))
  if args.shuffle is True:
    random.shuffle(all_data_tuples)
  with open(output_path, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for _, data_tuple in enumerate(all_data_tuples):
      csv_writer.writerow(list(data_tuple))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, choices=['train', 'val', 'test'],
                      default='train',
                      help='The path to the images from domain_a.')
  parser.add_argument('-n', '--dataset_name', type=str,
                      default='horse2zebra',
                      help='The name of the dataset in cyclegan_dataset.')
  parser.add_argument('-s', '--shuffle', type=bool,
                      default=True,
                      help='Whether to shuffle images when creating the dataset.')
  ARGS = parser.parse_args()
  create_dataset(ARGS)
