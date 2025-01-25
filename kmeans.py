#!/usr/bin/env python3

# Braeden Treutel
# CSCI 4350
# Intro to ai
# OLA 4

import sys
import numpy as np

class KMeans:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit(self, training_features, training_labels):
        # pick random centroids from the data (automatically randomized with split.bash)
        self.centroids = np.copy(training_features[:self.num_clusters])

        # initialize assignments and prev assignments with different values so the while loop executes
        assignments = np.zeros(training_features.shape[0])
        prev_assignments = np.full(training_features.shape[0], -1)
        
        # keep going until the assignments stay the same
        while not np.array_equal(assignments, prev_assignments):
            
            prev_assignments = assignments.copy()
            
            # calculate euclidean distances between every point and each centroid
            # use this to determine which distance is lowest, giving you your assignment for that point
            # (break ties by choosing lower cluster number)
            assignments = euclidean_distance(self.centroids, training_features)
    
            # average each of the points in each assignment to calculate the new centroid point for each cluster
            for i in range(self.num_clusters):
                # find the points that are assigned to a particular cluster
                assigned_points = training_features[assignments == i]
                # calculate the new centroids
                # make sure there is at least one point per centroid
                if (len(assigned_points) > 0):
                    self.centroids[i] = np.mean(assigned_points, axis=0)

        # majority vote for class label of each cluster
        # (prefer the smallest class label)
        self.cluster_labels = np.zeros(self.num_clusters)
        for i in range(self.num_clusters):
            # find the labels of the points that are assigned to a particular cluster
            assigned_labels = training_labels[assignments == i]

            # make sure that its not an empty cluster
            if len(assigned_labels) > 0:
                # return_counts will return the counts of each unique value in an array
                unique, counts = np.unique(assigned_labels, return_counts=True)
                # find the class label with the highest count
                majority = unique[np.argmax(counts)]
    
                self.cluster_labels[i] = majority
            else:
                self.cluster_labels[i] = -1

def loadData(fileName):
    data = np.loadtxt(fileName)
    # Special case of just one data sample will fail
    # without this check!
    if len(data.shape) < 2:
        data = np.array([data])
    # separate class labels from features
    features = data[:, :-1]
    class_labels = data[:, -1]
    
    return features, class_labels

# compare every point to the centroids and return their assignments
def euclidean_distance(centroids, data):
    # make an array that is (numoffeatures x numofcentroids)
    distances = np.zeros((data.shape[0], centroids.shape[0]))

    # iterate over all data points
    for i in range(data.shape[0]):
        # iterate over all centroids
        for j in range(centroids.shape[0]):
            # calculate euclidean distance for each point
            sqr_difference = (data[i] - centroids[j])**2
            sum_squared_difference = np.sum(sqr_difference)
            distance = np.sqrt(sum_squared_difference)
            distances[i,j] = distance

    # get the assignments
    # find minimum value across the columns for each row of the array
    assignments = np.argmin(distances, axis=1)
    return assignments


def main():

    # take the number of clusters from command-line
    num_clusters = int(sys.argv[1])
    
    # take the files from command-line
    training_file = sys.argv[2]
    validation_file = sys.argv[3]

    # load data into features and class labels
    training_features, training_labels = loadData(training_file)
    validation_features, validation_labels = loadData(validation_file)

    clusters = KMeans(num_clusters)
    clusters.fit(training_features, training_labels)

    # assign validation examples to a cluster from the training data
    assignments = euclidean_distance(clusters.centroids, validation_features)

    # compare assignments with true labels
    count = 0
    for i in range(len(validation_labels)):
        if validation_labels[i] == clusters.cluster_labels[assignments[i]]:
            count+=1

    print(count)
    
    
if __name__ == "__main__":
    main()