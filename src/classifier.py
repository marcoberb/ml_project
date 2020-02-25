"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from tabulate import tabulate
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if args.mode == 'TRAIN':
                    dataset = train_set
                elif args.mode == 'CLASSIFY':
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if args.mode == 'TRAIN':
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10]}
                svc = SVC(kernel='linear', probability=True)
                modeldue = GridSearchCV(svc, parameters)
                modeldue.fit(emb_array, labels)
                print("prova")
                print(modeldue.cv_results_)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

                # Saving training set to file
                create_training_set_file(paths, args.output_file)

            elif args.mode == 'CLASSIFY':
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)

                # Saving results to file
                create_result_table(class_names, paths, predictions, labels, args.output_file)


# Save to file a table containing used training set
def create_training_set_file(paths, filename):

    img_names = []

    for p in paths:
        img_names.append(p.split('/')[-1])

    table = tabulate({"img_name": img_names}, tablefmt="github")

    print("Printing training set on textfile")
    file = open(filename, "w+")
    file.write("IMG USED AS TRAINING SET:\n")
    file.write(table)
    file.close()


# Save to file a table of 'predictions' organized by 'class_names' for each img specified in 'paths'
def create_result_table(class_names, paths, predictions, labels, filename):

    indexes = []
    rows = []

    for p in paths:
        indexes.append(p.split('/')[-1])

    best_class_indices = np.argmax(predictions, axis=1)

    for pred, i in zip(predictions, range(0, len(best_class_indices))):
        row = np.append(pred, class_names[best_class_indices[i]])
        rows.append(row)

    table = rows
    headers = ["img_name"] + class_names + ["predicted_class"]
    formatted_table = tabulate(table, headers, tablefmt="github", showindex=indexes)

    accuracy = np.mean(np.equal(best_class_indices, labels))
    accuracy = "\n\nAccuracy: %.3f" % accuracy

    print("Printing results on textfile")
    file = open(filename, "w+")
    file.write(formatted_table)
    file.write(accuracy)
    file.close()


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification model should be used for classification', default='CLASSIFY')

    parser.add_argument('data_dir', type=str, help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. For training this is the output and for classification this is an input.')

    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')

    parser.add_argument('--test_data_dir', type=str, help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int, help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int, help='Use this number of images from each class for training and the rest for testing',
                        default=10)
    parser.add_argument('--output_file', type=str, help='Path to file where will be saved training set instances (when TRAIN mode) or classification '
                                                        'results (when CLASSIFY mode)', default='./results.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
