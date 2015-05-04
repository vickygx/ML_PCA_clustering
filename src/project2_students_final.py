import csv
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import matplotlib.cm as cm
import random


# returns the feature set in a numpy ndarray
def loadCSV(filename):
    stuff = np.asarray(np.loadtxt(open(filename, 'rb'), delimiter=','))
    return stuff


# returns list of artist names
def getArtists(directory):
    return [name for name in os.listdir(directory)]


# loads all image files into memory
def loadImages():
    image_files = [f for f in listdir('../artworks_ordered_50') if f.endswith('.png')]
    images = []
    for f in image_files:
        images.append(mpimg.imread(os.path.join('../artworks_ordered_50', f)))
    return images

        
# convert color image to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


# creates a feature matrix using raw pixel values from all images, one image per row
def loadPixelFeatures():
    images = loadImages()
    X = []
    for img in images:
        img = rgb2gray(img)
        img = img.flatten()
        X.append(img)
    return np.array(X)


def ml_compute_eigenvectors_SVD(X,m):
    left, s, right = np.linalg.svd(np.matrix(X))    
    U = np.matrix.getA(right)    
    return (U[0:m])


#Colour function: helper function for plot_2D_clusters
def clr_function(labels):
    colors = []
    for i in range(len(labels)):
        if(labels[i] == 0):
            color = 'red'
        elif(labels[i] == 1):
             color = 'blue'
        elif(labels[i] == 2):
            color = 'green'
        elif(labels[i] == 3):
            color = 'yellow'
        elif(labels[i] == 4):
            color = 'orange'
        elif(labels[i] == 5):
            color = 'purple'
        elif(labels[i] == 6):
            color = 'greenyellow'
        elif(labels[i] == 7):
            color = 'brown'
        elif(labels[i] == 8):
            color = 'pink'
        elif(labels[i] == 9):
            color = 'silver'
        else:
            color = 'black'                
        colors.append(color)
    return colors


#Plot clusters of points in 2D
def plot_2D_clusters(X, clusterAssignments, cluster_centers):    
    
    points = X
    labels = clusterAssignments
    centers = cluster_centers
            
   # points = X.tolist()
   # labels = clusterAssignments.tolist()
   # centers = cluster_centers.tolist()
                                            
    N = len(points)
    K = len(centers)
    x_cors = []
    y_cors = []
    for i in range(N):
        x_cors.append( points[i][0] )
        y_cors.append( points[i][1] )
            
    plt.scatter(x_cors[0:N], y_cors[0:N], color = clr_function(labels[0:N]))                    
    plt.title('2D toy-data clustering visualization')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')    

    x_centers = [0]* K
    y_centers = [0]* K    
    for j in range(K):
        x_centers[j] = centers[j][0]
        y_centers[j] = centers[j][1]
        
    plt.scatter(x_centers, y_centers, color = 'black', marker = ',')
    plt.grid(True)
    plt.show()
    return


#Plot original and reconstructed points in 2D 
def plot_pca(X_original, X_recon):
    x_orig = []
    y_orig = []
    x_cors = []
    y_cors = []
    for i in range(len(X_original)):
        x_orig.append( X_original[i][0] )
        y_orig.append( X_original[i][1] )        
        x_cors.append( X_recon[i][0] )
        y_cors.append( X_recon[i][1] )                
    plt.title('2D toy-data PCA visualization')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')   
    plt.axis('equal')   #Suggestion: Try removing this command and see what happens!
                            
    plt.scatter(x_orig, y_orig, color = 'red' )    
    plt.scatter(x_cors, y_cors, color = 'green', marker = ',')        
    plt.grid(True)    
    plt.show()
    return            


# display paintings by artist, one artist per matplotlib figure
def plotArtworks():
    artists = getArtists('../selected_subset')
    figure_count = 0
    for artist in artists:
        artist_dir = os.path.join('../', 'selected_subset', artist)
        image_files = [f for f in listdir(artist_dir) if f.endswith('.png')]
        print image_files
        n_row = math.floor(math.sqrt(len(image_files)))
        n_col = math.ceil(len(image_files)/n_row)
        fig = plt.figure(figure_count)
        fig.canvas.set_window_title(artist)
        for i in range(len(image_files)):
            plt.subplot(n_row, n_col,i+1)
            plt.axis('off')
            plt.imshow(mpimg.imread(os.path.join(artist_dir, image_files[i])))
        figure_count += 1
    plt.show()


# creates a dictionary mapping cluster label to indices of X that belong to that cluster
def create_cluster_dict(cluster_labels):
    clusters = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in clusters.keys():
            clusters[cluster_labels[i]] = [i]
        else:
            clusters[cluster_labels[i]].append(i)
    return clusters


# plots clusters of images
def plotClusters(cluster_labels):
    ordered_artist_dir = os.path.join('../', 'artworks_ordered_50')
    image_files = [f for f in listdir(ordered_artist_dir) if f.endswith('.png')]
    clusters = create_cluster_dict(cluster_labels)
    figure_count = 0
    for key in clusters.keys():
        n_row = math.floor(math.sqrt(len(clusters[key])))
        n_col = math.ceil(len(clusters[key])/n_row)
        fig = plt.figure(figure_count)
        fig.canvas.set_window_title(str(key))
        for i in range(len(clusters[key])):
            plt.subplot(n_row, n_col, i+1)
            plt.axis('off')
            plt.imshow(mpimg.imread(os.path.join(ordered_artist_dir, image_files[clusters[key][i]])))
        figure_count += 1
    plt.show()


# displays images specified in labeled.csv after reconstruction (grayscale)
# Input:
    # matrix of pixel values, one image per row
# Output:
    # plot of the selected images in labeled.csv
def plotGallery(reconstruction):
    ordered_artist_dir = os.path.join('../', 'artworks_ordered_50')
    image_files = [f for f in listdir(ordered_artist_dir) if f.endswith('.png')]
    indices = loadCSV('labeled.csv')
    num_images = len(indices)
    n_row = math.floor(math.sqrt(num_images))
    n_col = math.ceil(num_images/n_row)
    for i in range(indices.shape[0]):
        plt.subplot(10, 5, i+1)
        plt.axis('off')
        img = np.reshape(reconstruction[int(indices[i])-1], (50,50))
        plt.imshow(img, cmap=cm.gray)
    plt.show()


# computes the majority vote for each cluster
# Input:
    # list of labels of each row in the feature matrix
    # fraction of data points that are labeled, defaults to 1 (all points have labels)
# Output:
    # a dictionary, with (key, value) = (cluster_label, majority)
def majorityVote(cluster_labels, labeled = 100):
    artist_labels = loadCSV('artist_labels_' + str(labeled) + '.csv')
    clusters = create_cluster_dict(cluster_labels)
    majorities = {} 
    for key in clusters.keys():
        votes = []
        for i in range(len(clusters[key])):
            label = artist_labels[clusters[key][i]]
            if label != -1:
                votes.append(label)
        if len(votes) == 0:
            votes.append(-1)
        votes = np.array(votes)
        majorities[key] = stats.mode(votes)[0][0]
    return majorities


# returns the total number of classification errors, comparing the majority vote label to true label
def computeClusterPurity(cluster_labels, majorities=None):
    if majorities == None:
        majorities = majorityVote(cluster_labels)
    artist_labels = loadCSV('artist_labels.csv')
    clusters = create_cluster_dict(cluster_labels)
    errors = 0 
    for key in clusters.keys():
        majority = majorities[key]
        for i in range(len(clusters[key])):
            if artist_labels[clusters[key][i]] != majority:
                errors += 1
    return 1-(float(errors)/float(len(cluster_labels)))


# computes the majority vote for each cluster
# Input:
    # list of labels of each row in the feature matrix
    # fraction of data you have labeled (valid inputs: 5, 15, 25, 50, 75, 100)
# Output:
    # classification accuracy
def classifyUnlabeledData(cluster_labels, labeled):
    majorities = majorityVote(cluster_labels, labeled)
    acc = computeClusterPurity(cluster_labels, majorities)
    return acc

# computes the maximum pairwise distance within a cluster
def intraclusterDist(cluster_values):
    max_dist = 0.0 
    for i in range(len(cluster_values)):
        for j in range(len(cluster_values)):
            dist = np.linalg.norm(cluster_values[i]-cluster_values[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


# helper function for Dunn Index
def index_to_values(indices, dataset):
    output = []
    for index in indices:
        output.append(dataset[index])
    return np.matrix(output)


# computes the Dunn Index, as specified in the project description
# Input:
    # cluster_centers - list of cluster centroids
    # cluster_labels - list of labels of each row in feature matrix
    # features - feature matrix 
# Output:
    # dunn index (float)
def computeDunnIndex(cluster_centers, cluster_labels, features):  
    clusters = create_cluster_dict(cluster_labels)
    index = float('inf')  
    max_intra_dist = 0.0
    # find maximum intracluster distance across all clusters
    for i in range(len(cluster_centers)):
        cluster_values = index_to_values(clusters[i], features)
        intracluster_d = float(intraclusterDist(cluster_values))
        if intracluster_d > max_intra_dist:
            max_intra_dist = intracluster_d

    # perform minimization of ratio
    for i in range(len(cluster_centers)):
        inner_min = float('inf')
        for j in range(len(cluster_centers)):
            if i != j:
                intercluster_d = float(np.linalg.norm(cluster_centers[i]-cluster_centers[j]))
                ratio = intercluster_d/max_intra_dist
                if ratio < inner_min:
                    inner_min = ratio
        if inner_min < index:
            index = inner_min
    return index

#helper function for init_medoids_plus
def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i

#helper function for init_medoids_plus
def square_distance(point1, point2):
    value = 0.0    
    for i in range(0,len(point1)):
        value += (point1[i] - point2[i])**2    
    return value

#Function for generating initial centers uniformly at random (without replacement) from the data
def init_medoids(X, K):        
    points = X
    N = len(points)
    indices = range(0,N)
    centers = list([0]*K)
        
    for j in range(0,K):            
        temp = random.randrange(0,N-j)       
        centers[j] = indices[temp]
        del indices[temp]    

    medoids = []        
                        
    for j in range(0,K):
        medoids.append(points[centers[j]])
    
    return medoids


#Function for generating initial centers according to the KMeans++ initializer
#This is a faster version
def init_medoids_plus(X, K):        
    points = X
    N = len(points)
    indices = range(0,N)
    medoids = []
    weights = []
    
    #initialize first medoid
    temp = random.randrange(0,N)       
    medoids.append(list(points[indices[temp]]))
    del indices[temp]
    
    for i in range(len(indices)):
        weights.append(square_distance(medoids[0],points[indices[i]]))        
            
    
    for j in range(0,K):            
        if(j == 0):
            continue
               
        for i in range(len(indices)):
            c = medoids[j-1]
            if(square_distance(c, points[indices[i]]) < weights[i]):
                weights[i] = square_distance(c, points[indices[i]])
                                        
        temp = weighted_choice(weights)
        medoids.append(list(points[indices[temp]]))    
        del indices[temp]
        del weights[temp]
         
    #return np.array(medoids)
    return medoids
            
            
# Centers data via mean subtraction
# Input:
    # X - n X d numpy array [n data points, d features]
# Output: 
    # (X_centered, mean)
    # X_centered[i,j] = X[i,j] - mean[j]
    # mean = (d x 1 ) numpy array where mean[j] is the mean of all the 
        # values in the jth column 
        # for each column (i.e. each feature)
def ml_split(X):
    mean = X.mean(0)
    X_centered = np.subtract(X, mean)
    return (X_centered, mean)
    
# Computes principal component directions of given matrix
# Input:
    # X - n x d numpy array of n data points, d features
    # m - number of pinciple components to compute
# Output:    
    # U - m x d matrix (rows are the first m PCA directions)
def ml_compute_eigenvectors(X,m):
    # computes covariance matrix
    covariance = np.dot(X.transpose(), X)
    # finds eigenvalues and vectors
    eig_values, eig_vectors = np.linalg.eigh(covariance)

    # turn eigenvector column matrix into eigenvector row matrix
    eig_vectors = eig_vectors.transpose()
    
    # sort eigenvectors by decreasing eigenvalues and return top m eigenvectors
    sorted_idx = eig_values.argsort()[::-1][:m]
    
    return eig_vectors[sorted_idx]

# Implements PCA dimensionality reduction of a (centered) data matrix X
# Input:
#       X - n x d numpy array of n data points, d features
#       U - m x d matrix whose rows are the top m eigenvectors of X^T*X, in decending order of eigenvalues 
#           with eigenvectors normalized
# Output:
#       E - n x m matrix, whose rows represent low-dimensional feature vectors
def ml_pca(X, U):
    # Project our original d-dimensional feature vectors (xi) to k-D vectors and return
    return np.dot(X, U.transpose())

# Reconstructs original data using the PCA features and eigenvectors
# Input:
#       E - n x m numpy array of n data points, with m features
#       U - m x d matrix whose rows are the top m eigenvectors of X^T*X
#       mean - vector that is the mean of the data
# Output:
#       X_recon - n x d numpy array of n reconstructed data points
def ml_reconstruct(U, E, mean):
    # Get original X from E and U (TODO not sure)
    X = np.dot(E, U)
    
    # add the mean array back to X to get the original X since all the X's were centered via mean subtraction.
    return np.add(X, mean)

# Helper function for k means and k medoids to update clusters to closest centroid
# returns (clusters, clusterAssignments)
def update_clusters(X, centroids):
        clusterAssignments = np.empty([len(X), 1])
        clusters = [[] for i in range(len(centroids))]  

        dist_matrix = distance.cdist(X, centroids, 'sqeuclidean')
        # Assign each point to the cluster of the closest point
        for i in range(len(X)):
            closest_cluster = np.argmin(dist_matrix[i])
            clusterAssignments[i] = closest_cluster
            clusters[closest_cluster].append(i)

        return (clusters, clusterAssignments)

# Implements k means clustering
# Input:
#       X - n x d numpy array of n data points, with d features
#       k - number of desired clusters
#       init - k x d matrix of k data points, with d features (initial guesses for centroids)
# Output:
#       (centroids, clusterAssignments)
#       centroids - k x d matrix of k data points, with d features (final centroids)
#       clusterAssignmetns - n x 1 numpy array, each position in (0, k - 1), 
#           inidicating which cluster the input points belong to
def ml_k_means(X, K, init):
    clusterAssignments = np.empty([len(X), 1])
    centroids = np.copy(init)

    # Repeat until there is no further change in cost
    while(True):
        # --Update clusters
        clusters, clusterAssignments = update_clusters(X, centroids)

        has_changed = False

        # --Update centroids by finding new cluster mean
        for c in range(K):
            if len(clusters[c]) != 0:
                cluster = X[clusters[c]]
                new_centroid = np.mean(cluster, axis=0)
                
                if not np.array_equal(centroids[c], new_centroid):
                    centroids[c] = new_centroid
                    has_changed = True
        
        if not has_changed:
            break


    return (centroids, clusterAssignments)

# Implements k medoids clustering
# Input:
#       X - n x d numpy array of n data points, with d features
#       k - number of desired clusters
#       init - k x d matrix of k data points, with d features (initial guesses for medoids)
# Output:
#       (centroids, clusterAssignments)
#       medoids - k x d matrix of k data points, with d features (final medoids)
#       clusterAssignmetns - n x 1 numpy array, each position in (0, k - 1), 
#           inidicating which cluster the input points belong to
def ml_k_medoids(X, K, init):    
    clusterAssignments = np.empty([len(X), 1])
    medoids = np.copy(init)

    # k lists containing a list of indices of the points in that cluster
    clusters = [[] for i in range(K)]     

    # Repeat until there is no further change in cost
    while(True):
        # --Update clusters
        clusters, clusterAssignments = update_clusters(X, medoids)

        has_changed = False

        # -- For each cluster, update medoids by finding the point that minimizes the cost
        for c in range(K):
            cluster_point_indices = clusters[c]

            # Keep the cluster center the same if no points are given to it
            if len(cluster_point_indices) == 0:
                continue

            cluster = X[cluster_point_indices]
            dist_matrix = distance.cdist(cluster, cluster, 'sqeuclidean')
            cost_array = dist_matrix.sum(axis=1)
            next_medoid_index = cost_array.argmin(axis=0)
            new_medoid = np.copy(cluster[next_medoid_index])

            if not np.array_equal(medoids[c], new_medoid):
                medoids[c] = new_medoid
                has_changed = True
            

        if not has_changed:
            break

    return (medoids, clusterAssignments)


def main():
    # --------------------- Problem 1 -----------------------
    if (False):  
        # Finds the best projection of a set of 2D points on a line
        m = 1
        # Reads in toy_pca_data.csv
        X = loadCSV('toy_pca_data.csv')
        # Generate means vector and center the data
        X_centered, mean = ml_split(X)
        # generate top m = 1 eigenvectors
        U = ml_compute_eigenvectors(X_centered, m)
        # generate low-dimensional feature vectors
        E = ml_pca(X_centered, U)
        # reconstruct original points using low-d feature vectors
        X_recon = ml_reconstruct(U, E, mean)
        # plot original points together with reconstructed points on a single scatter plot 
        plot_pca(X, X_recon)

    # --------------------- Problem 2 A ----------------------
    if (False):  
        # Reads in toy_pca_data.csv
        X = loadCSV('toy_cluster_data.csv')
        for k in [2,3,4]:
            init = X[:k]
            cluster_centers, clusterAssignments = ml_k_means(X, k, init)
            plot_2D_clusters(X, clusterAssignments, cluster_centers)

    # --------------------- Problem 2 B ----------------------
    if (False):  
        # Reads in toy_pca_data.csv
        X = loadCSV('toy_cluster_data.csv')
        for k in [2,3,4]:
            init = X[:k]
            cluster_centers, clusterAssignments = ml_k_medoids(X, k, init)
            plot_2D_clusters(X, clusterAssignments, cluster_centers)
     
    # --------------------- Testing 2   ---------------------- 
    if (False):
        X = np.array([[0.,0.], [6.,6.], [5.,5.], [10.,10.], [6.,7.], [4.,6.], [2.,1.], [3.,2.]])
        for k in [2,3,4]:
            init = X[:k]
            cluster_centers, clusterAssignments = ml_k_medoids(X, k, init)
            plot_2D_clusters(X, clusterAssignments, cluster_centers)
        
    # --------------------- Problem 3   ----------------------
    if (False):
        plotArtworks()

    # --------------------- Problem 4/5 ----------------------
    if (False):
        m = 40
        # raw grayscale pixel values for an image, each on a row
        X = loadPixelFeatures()
        X_centered, mean = ml_split(X)
        U = ml_compute_eigenvectors_SVD(X_centered, m)
        E = ml_pca(X_centered, U)

        X_recon = ml_reconstruct(U, E, mean)

        plotGallery(X_recon)

    # --------------------- Problem 6   ----------------------
    if (False):
        X = loadPixelFeatures()
        X_centered, mean = ml_split(X)
        for m in [10, 20, 50, 100, 400]:
            print "M: ", m
            # perform PCA on X
            U = ml_compute_eigenvectors_SVD(X_centered, m)
            # reconstruct data matrix using only m dimensions to produce a new data matrix E
            E = ml_pca(X_centered, U)
            
            # run K-means function on E for different values of k
            for k in [2,5,10,11]:
                print "K: ", k
                init = init_medoids_plus(E, k)
                cluster_centers, clusterAssignments = ml_k_means(E,k,init)
                clusterAssignments = [assignment[0] for assignment in clusterAssignments.tolist()]

                # plotClusters(clusterAssignments)

                print computeDunnIndex(cluster_centers, clusterAssignments, E)
                print computeClusterPurity(clusterAssignments, majorities=None)

    # --------------------- Problem 7 ------------------------
    if (False):
        # Seeing results for different features types and k
        m = 200

        X_S = []
        # Raw Pixel Features
        X_S.append(loadPixelFeatures())
        X_S.append(loadCSV('gist_features.csv'))
        X_S.append(loadCSV('deep_features.csv'))
        
        U_S = []
        E_S = []
        for X in X_S:
            X_centered, mean = ml_split(X)
            U = ml_compute_eigenvectors_SVD(X_centered, m)
            U_S.append(U)
            E = ml_pca(X_centered, U)
            E_S.append(E)

        for i,E in enumerate(E_S):
            print "E: i:", i
            for k in [2,5,10]:
                print "K: ", k

                init = init_medoids_plus(E, k)
                cluster_centers, clusterAssignments = ml_k_means(E,k,init)
                clusterAssignments = [assignment[0] for assignment in clusterAssignments.tolist()]

                plotClusters(clusterAssignments)

                print computeDunnIndex(cluster_centers, clusterAssignments, E)
                print computeClusterPurity(clusterAssignments, majorities=None)

    # --------------------- Problem 8 ------------------------
    if (True):
        m = 200
        X = loadCSV('deep_features.csv')
        X_centered, mean = ml_split(X)
        U = ml_compute_eigenvectors_SVD(X_centered, m)
        E = ml_pca(X_centered, U)
  
        for k in [10,30,50,70]:
            print "K:", k
            for percent in [5, 15, 25, 50, 75, 100]:
                print "Percent:", percent
                accuracy = 0
                for i in range(3):
                    init = init_medoids_plus(E, k)
                    cluster_centers, clusterAssignments = ml_k_means(E, k, init)
                    clusterAssignments = [assignment[0] for assignment in clusterAssignments.tolist()]

                    accuracy += classifyUnlabeledData(clusterAssignments, percent)

                print accuracy/3


main()
    