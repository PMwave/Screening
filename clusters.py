import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

with open('X_embedded.npy', 'rb') as f:
    X_embedded = np.load(f)

with open('y_pred.npy', 'rb') as f:
    y_pred = np.load(f)
    


# sns settings
sns.set(rc={'figure.figsize':(13,9)})

# colors
palette = sns.hls_palette(450, l=.4, s=.9)

# plot
fig = plt.figure()
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title('t-SNE with Kmeans Labels')
st.pyplot(fig)
plt.savefig("improved_cluster_tsne.png")
plt.show()
