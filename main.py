import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the customer data into a pandas DataFrame
df = pd.read_csv("customer_data.csv")

# Extract the customer spending data into a numpy array
X = df[["Spending_Score"]].values

# Fit the KMeans model to the data
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points and centroids
plt.scatter(X[:, 0], np.zeros(X.shape[0]), c=labels)
plt.scatter(centroids[:, 0], np.zeros(centroids.shape[0]), marker='x', s=200, linewidths=3, color='r')
plt.show()
