# UFO Sightings in different countries

import numpy as np
import pandas as pd
ufo_data=pd.read_csv(r'C:\Users\shreya\Desktop\scrubbed.csv', low_memory=False)
ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'],errors='coerce')

x=ufo_data[['longitude','latitude']].dropna()
x=x.to_numpy()

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
kmeans.fit(x)
kmeans.cluster_centers_
ypred=kmeans.labels_
import matplotlib.pyplot as plt
plt.scatter(x[:,0],x[:,1],c=kmeans.labels_ )

