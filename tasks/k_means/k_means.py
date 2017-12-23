from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float

image = imread('data/parrots.jpg')
image = img_as_float(image)

# transform image array into X-matrix
X = image.reshape(-1, 3)

model = KMeans(init='k-means++', random_state=241)
model.fit(X)

cluster_labels = model.predict(X)
n_clusters = list(range(1,9))




submission_file = open('submissions/knn-metric-tuning/best_p.txt', 'w+')
submission_file.write(str( numpy.round(best_p, 2) ))
submission_file.close()

print( numpy.round(best_p, 2) )
