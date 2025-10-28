#import all libarires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load dataset
data = pd.read_csv("/home/spotify dataset.csv")
data['track_name'].fillna('unknown',inplace=True)
data['track_artist'].fillna('unknown',inplace=True)
data['track_album_name'].fillna('unknown',inplace=True)
data.drop_duplicates(inplace=True)

#scaling the data
features = ['danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness',
'valence','tempo','duration_ms']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

sns.histplot(data['energy'],kde=True)
plt.title('Energy Distribution')
plt.show()

sns.boxplot(x='playlist_genre',y='danceability',data=data)
plt.xticks(rotation=45)
plt.title('Danceability by Genre')
plt.show()

plt.figure(figsize=(11,5))
sns.heatmap(data[features].corr(),annot=True,cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

X = data[features]
kmeans = KMeans(n_clusters=3,random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

pca=PCA(n_components=2)
pca_data=pca.fit_transform(X)
data["PCA1"]=pca_data[:,0]
data["PCA2"]=pca_data[:,1]

sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='playlist_genre', style='Cluster', s=5)
plt.title("Song Clusters by Genre")
plt.show()

def recommend(song):
    song = song.lower()
    if song not in data['track_name'].str.lower().values:
        print("Song not found!")
        return
    idx = data[data['track_name'].str.lower() == song].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(f"\n🎵 Songs similar to '{song}':\n")

    for i in range(1,6):
        s = data.iloc[scores[i][0]]
        print(f"{i}. {s['track_name']} - {s['track_artist']} ({s['playlist_genre']})")