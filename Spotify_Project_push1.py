#!/usr/bin/env python
# coding: utf-8

# # # Step 1 Data preparation

# In[142]:


get_ipython().system('pip install spotipy')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install yellowbrick')


# In[59]:





# In[60]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler


from yellowbrick.target import FeatureCorrelation
from scipy.stats import norm
from scipy import stats


import warnings
warnings.filterwarnings("ignore")
color = sns.color_palette()
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# In[62]:


# Data Collection, Data Cleaning & Data Manipulation 
import numpy as np 
import pandas as pd 

import wordcloud

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Data Transformation
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats


from yellowbrick.target import FeatureCorrelation
from scipy.stats import norm
from scipy import stats


import warnings
warnings.filterwarnings("ignore")
color = sns.color_palette()
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# In[177]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

idcl ="8430dac4c0cc41d580798dc2b97b4cd5" 
clsec = "22dd2a849bbc4e40b63c7bf4f7d353f5"

client_credentials_manager = SpotifyClientCredentials(client_id=idcl, client_secret=clsec)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[178]:


## First we run a self search
artist_names = ["Yash Vaid"]
artist_ids = []
for artist_name in artist_names:
    results = sp.search(q='artist:' + artist_name,
type='artist')
    
    if len(results['artists']['items']) > 0:
        artist_id = results['artists']['items'][0]['id']
        artist_ids.append(artist_id)
    else:
        print(f"No artist found with name {artist_name}")

print(artist_ids)


# Now we get the top 30 tracks for yash 

# In[179]:


yashtracks = []
results = sp.search(q='track:Tune kaha (Reprise)', type='track')
if results['tracks']['total'] > 0:
    track = results['tracks']['items'][0]
    track_name = track['name']
    track_uri = track['uri']
    track_id = track['id']
    track_popularity = track['popularity']
    artist_id = track['artists'][0]['id']
    artist_name = track['artists'][0]['name']
    #retrieve all tracks associated with the artist
    artist_tracks = sp.artist_top_tracks(artist_id)
    for track in artist_tracks['tracks']:
        yashtracks.append({'name': track['name'], 'id': track['id'], 'artist_name': artist_name, 'artist_id': artist_id,
                       'track_uri': track_uri, 'track_id': track_id, 'track_popularity': track_popularity})
else:
    print("Track not found.")

print(yashtracks)


# In[ ]:





# The using names and ids of tracks we get the features of each track

# In[180]:


yash_tracks_with_features = []
for track in yashtracks:
    track_id = track['id']
    audio_features = sp.audio_features(track_id)
    track.update(audio_features[0])
    yash_tracks_with_features.append(track)
print(yash_tracks_with_features)


# Then we convert it into a dataframe to be converted as a csv file 

# In[181]:




yash_tracks_df = pd.DataFrame(yash_tracks_with_features)
yash_tracks_df.to_csv('yashtracks.csv',index=False)


# Now take the list of artists i want to compare my songs with which will also be our dataset for model building

# In[ ]:





# In[182]:


## Full Search
artist_names = ["Sleepy Fish","Jeremy Zucker","Shallou","Frank Ocean","EDEN","Purrple Cat","PinkPantheress","San Holo","Keshi","OneHeart","Honey Ivy","Philantrope","Ben Bohmer","Vaid Yash"]
artist_ids = []
tracks = []

for artist_name in artist_names:
    results = sp.search(q='artist:' + artist_name, type='artist')
    if len(results['artists']['items']) > 0:
        artist_id = results['artists']['items'][0]['id']
        artist_ids.append(artist_id)
        artist_tracks = sp.artist_top_tracks(artist_id)
        for track in artist_tracks['tracks']:
            tracks.append({'name': track['name'], 'id': track['id'], 'artist_name': artist_name, 'artist_id': artist_id,
                           'track_uri': track['uri'], 'track_popularity': track['popularity']})
    else:
        print(f"No artist found with name {artist_name}")

print(artist_ids)
print(tracks)
print


# Similarly we get their features by using their track ids in a list format 

# In[183]:


tracks_with_features = []
for track in tracks:
    track_id = track['id']
    audio_features = sp.audio_features(track_id)
    track.update(audio_features[0])
    tracks_with_features.append(track)
print(tracks_with_features)


# The list obtained is finally converted into a data frame as well a csv, file 

# In[184]:


import pandas as pd

tracks_df = pd.DataFrame(tracks_with_features)
tracks_df.to_csv('tracks.csv',index=False)

all_tracks = tracks_df.append(yash_tracks_df, ignore_index=True)
Final_data = pd.DataFrame(all_tracks)
Final_data.to_csv('Final_Data_Tracks.csv',index=False)
print(Final_data)


# # Track Search

# In[185]:


is_present = "Rukhsat" in [track['name'] for name in Final_data]
if is_present:
    print("Track is present in the list.")
else:
    print("Track is not present in the list.")


# This completes our data preparation part 

# # EDA

# In[176]:


print(Final_data)


# In[186]:


data = pd.read_csv(r'C:\Users\yjain\Final_Data_Tracks.csv', header=0)


# In[187]:


data.shape


# In[188]:


data.info


# In[189]:


data.head


# In[190]:


data.describe().transpose()


# In[11]:


data.hist(bins = 20, color = 'orange', figsize = (20, 14))


# In[11]:



sns.set_style('darkgrid')
sns.set_palette('deep', color_codes=True)

data.hist(bins = 20, color = 'orange', figsize = (20, 14), edgecolor = 'black')
sns.despine(left=True, bottom=True)


# In[18]:


df = data.copy()


# In[19]:


df.isnull().sum()


# In[20]:


df.duplicated().sum()


# In[21]:


#FInd the categorical variables 
categorical_df = df.select_dtypes(include = 'object')

categorical_df.info()


# In[22]:


#CHeck Cardinality for col in categorical_df.columns:
for col in categorical_df.columns:
    print(f'{col}: {categorical_df[col].nunique()}')
    print('\n')


# In[23]:


#Find the most popular artists in data set
from wordcloud import WordCloud

plt.figure(figsize = (20, 14))

def visualize_word_counts(counts):
    wc = WordCloud(max_font_size=130, min_font_size=25, colormap='tab20', background_color='white', prefer_horizontal=.95, width=2100, height=700, random_state=0)
    cloud = wc.generate_from_frequencies(counts)
    plt.figure(figsize=(18,15))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[24]:


lead_artists = df['artist_name'].value_counts().head(20)

lead_artists


# In[25]:


fig, ax = plt.subplots(figsize = (12, 10))
sns.set_style("darkgrid")

sns.heatmap(pd.DataFrame(lead_artists), annot = True, cmap = 'RdBu', ax = ax)

ax.set_title('20 Most Popular Artists in Dataset', color = 'red', fontsize = 14, weight = 'bold')

plt.tight_layout()
plt.show()


# In[26]:


visualize_word_counts(lead_artists)


# In[27]:


most_popularity = df.query('track_popularity < 86', inplace = False).sort_values('track_popularity', ascending = False)

top_10_artists = most_popularity.head(10)

top_10_artists


# In[28]:


lead_songs = most_popularity[['name', 'track_popularity']].head(20)

lead_songs


# In[29]:


fig, ax = plt.subplots(figsize = (10, 10))

sns.barplot(x = lead_songs.track_popularity, y = lead_songs.name, palette = 'viridis', orient = 'h', edgecolor = 'black', ax = ax)

ax.set_xlabel('Popularity', c ='red', fontsize = 12, weight = 'bold')
ax.set_ylabel('Songs', c = 'red', fontsize = 12, weight = 'bold')
ax.set_title('20 Most Popular Songs in Dataset', c = 'red', fontsize = 14, weight = 'bold')

plt.show()


# In[30]:


yash_comparison = df.query('artist_name == "Yash Vaid"', inplace = False).sort_values('track_popularity', ascending = False)
print(yash_comparison)


# In[31]:


from sklearn import preprocessing

feat_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

mean_vals = pd.DataFrame(columns=feat_cols)
mean_vals = mean_vals.append(most_popularity[feat_cols].mean(), ignore_index=True)
mean_vals = mean_vals.append(yash_comparison[feat_cols].mean(), ignore_index=True)

print(mean_vals)

import plotly.graph_objects as go
import plotly.offline as pyo
fig = go.Figure(
    data=[
        go.Scatterpolar(r=mean_vals.iloc[0], theta=feat_cols, fill='toself', name='Top Songs'),
        go.Scatterpolar(r=mean_vals.iloc[1], theta=feat_cols, fill='toself', name='Yash Vaid Songs'),
    ],
    layout=go.Layout(
        title=go.layout.Title(text='Feature comparison'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
)

#pyo.plot(fig)
fig.show()


# In[32]:


#Finding songs by one feature
most_acousticness = df.sort_values(by='acousticness',ascending=False).head(10)

most_acousticness


# In[33]:


df.describe()


# In[34]:


num_df = df.select_dtypes(include = 'number')


# In[37]:


plt.style.use('seaborn')

names = list(num_df.columns)

plot_per_row = 2

f, axes = plt.subplots(round(len(names)/plot_per_row), plot_per_row, figsize = (15, 25))

y = 0;

for name in names:
    i, j = divmod(y, plot_per_row)
    sns.histplot(x=df[name], kde = True, ax=axes[i, j], color = 'blue')
    y = y + 1

plt.tight_layout()
plt.show()


# In[39]:


plt.style.use('seaborn')

names = list(num_df.columns)

plot_per_row = 2

f, axes = plt.subplots(round(len(names)/plot_per_row), plot_per_row, figsize = (15, 25))

y = 0;

for name in names:
    i, j = divmod(y, plot_per_row)
    sns.boxplot(x=df[name], ax=axes[i, j], palette = 'Set3')
    y = y + 1

plt.tight_layout()
plt.show()


# In[137]:


sns.pairplot(df,corner=True,hue='artist_name')


# In[41]:


plt.figure(figsize = (20, 14))

corr_matrix = df.corr()
cmap = sns.color_palette('viridis')
sns.heatmap(corr_matrix, annot = True, cmap = cmap)
plt.title('Correlation between numerical features')
plt.show()


# In[43]:


corr_matrix["track_popularity"].sort_values(ascending=False)


# In[45]:


from pandas.plotting import scatter_matrix

attributes = ["track_popularity", "loudness", "energy"]

scatter_matrix(df[attributes], figsize=(12, 8))

plt.show()


# In[49]:


import plotly.express as px


# In[50]:


px.box(data_frame=df,y='duration_ms',color='key')


# In[191]:


feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','key']

X, y = data[feature_names], data['track_popularity']

# Create a list of the feature names

features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()   


# In[192]:


total = data.shape[0]
popularity_score_more_than_40 = data[data['track_popularity'] > 50].shape[0]

probability = (popularity_score_more_than_40/total)*100
print("Probability of song getting more than 40 in popularity :", probability)


# In[193]:


features_o = ['liveness','valence','acousticness']

plt.rcParams['figure.figsize'] = (15, 4)

plt.subplot(1, 3, 1)
sns.distplot(data['liveness'])

plt.subplot(1, 3, 2)
sns.distplot(data['valence'])

plt.subplot(1, 3, 3)
sns.distplot(data['acousticness'])

plt.suptitle('Checking Feature with more correlation', fontsize = 10)
plt.show()


# In[73]:


plt.rcParams['figure.figsize'] = (15, 4)

plt.subplot(1, 3, 1)
res = stats.probplot(data['liveness'], plot=plt)

plt.subplot(1, 3, 2)
res = stats.probplot(data['valence'], plot=plt)

plt.subplot(1, 3, 3)
res = stats.probplot(data['acousticness'], plot=plt)


# In[75]:


#Univariate analysis to recognize values as outliers 
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(data['track_popularity'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[76]:


#Bivariate analysis

x = data.groupby("acousticness")["track_popularity"].mean().sort_values(ascending=False).head(20).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(x["acousticness"].values, x['track_popularity'].values, color=color[4])
plt.ylabel('acousticness', fontsize=12)
plt.xlabel('track_popularity', fontsize=12)
plt.title("popularity wise acousticness", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[82]:


fig1 = sns.jointplot(x="acousticness", y="track_popularity", data=data.sample(100),
                  kind="reg", truncate=False,
                  color=color[4])


# In[96]:


fig2 = sns.jointplot(x="liveness", y="track_popularity", data=data.sample(100),
                  kind="reg", truncate=False,
                  color=color[5])


# In[91]:


fig3 = sns.jointplot(x="key", y="track_popularity", data=data.sample(100),
                  kind="reg", truncate=False,
                  color=color[2])


# In[138]:


data_w_gen['key'].value_counts()


# In[97]:


data.describe


# In[194]:


from scipy.spatial import distance


# In[195]:


def make_matrix_correlation(data,song,number):
    data.drop_duplicates(inplace=True)
    songs=data['name'].values
#    best = difflib.get_close_matches(song,songs,1)[0]
    best=find_word(song,songs)
    print('The song closest to your search is :',best)
    genre=data[data['name']==best]['key'].values[0]
    df=data[data['key']==genre]
    x=df[df['name']==best].drop(columns=['key','name']).values
    if len(x)>1:
        x=x[1]
    song_names=df['name'].values
    df.drop(columns=['key','name'],inplace=True)
    df=df.fillna(df.mean())
    p=[]
    count=0
    for i in df.values:
        p.append([distance.correlation(x,i),count])
        count+=1
    p.sort()
    for i in range(1,number+1):
        print(song_names[p[i][1]])
    


# In[196]:


# This is a function to find the closest song name from the list
def find_word(word, words):
    words = np.array(words)
    t = []
    count = 0
    if word[-1] == ' ':
        word = word[:-1]
    for i in words:
        if word.lower() in i.lower():
            t.append([float(len(word) / len(i)), count])
        else:
            t.append([0, count])
        count += 1
    t.sort(reverse=True)
    return words[t[0][1]]


# In[197]:


e=input('Please enter The name of the song :')
f=int(input('Please enter the number of recommendations you want: '))
make_matrix_correlation(df,e,f)


# In[133]:





# In[201]:


data['key'].value_counts()


# In[208]:


df_movie_tunes = data[data['key'] == "['7']"]
df_show_tunes = data[data['key'] == "['0']"]
df_classical_piano = data[data['key'] == "['8']"]
df_sleep = data[data['key'] == "['9']"]


# In[ ]:





# In[218]:


# now lets check for top2 gener that is show tunes

fig,ax = plt.subplots(figsize=(20, 10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})

sns.lineplot(x="track_popularity", y="acousticness", data=data, color="b",label = 'acousticness')
sns.lineplot(x="track_popularity", y="liveness", data=data, color="r",label = 'liveness')
sns.lineplot(x="track_popularity", y="energy", data=data, color="y",label = 'energy')
sns.lineplot(x="track_popularity", y="valence", data=data, color="m",label = 'valence')

plt.rcParams["xtick.labelsize"] = 15

ax.set_title('Audio characteristics of genres "Show Tunes" data by the Key used')
ax.legend(fontsize = 14)


# In[222]:


# now lets check for top3 gener that is classical piano 
fig,ax = plt.subplots(figsize=(20, 10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth":3})

sns.barplot(x="key", y="acousticness", data=data, color="b", label = 'acousticness')
sns.barplot(x="key", y="danceability", data=data, color="r", label = 'danceability')
sns.barplot(x="key", y="speechiness", data=data, color="g", label = 'speechiness')
sns.barplot(x="key", y="energy", data=data, color="y", label = 'energy')
sns.barplot(x="key", y="valence", data=data, color="m", label = 'valence')

plt.rcParams["xtick.labelsize"] = 15

ax.set_title('Audio characteristics of most popular songs by the Key used')
ax.legend(fontsize=14)

# now lets check for top3 gener that is sleep betts 
# now lets check for top3 gener that is classical piano 
fig,ax = plt.subplots(figsize=(20, 10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth":3})

sns.barplot(x="key", y="acousticness", data=lead_songs, color="b", label = 'acousticness')
sns.barplot(x="key", y="danceability", data=lead_songs, color="r", label = 'danceability')
sns.barplot(x="key", y="speechiness", data=lead_songs, color="g", label = 'speechiness')
sns.barplot(x="key", y="energy", data=lead_songs, color="y", label = 'energy')
sns.barplot(x="key", y="valence", data=lead_songs, color="m", label = 'valence')

plt.rcParams["xtick.labelsize"] = 15

ax.set_title('Audio characteristics of most popular songs by the Key used')
ax.legend(fontsize=14)


# In[240]:


df_top1 = data['artist_name'] == "['Jeremy Zucker']"
df_top2 = data['artist_name'] == "['Yash Vaid']"
df_top3 = data['artist_name'] == "['EDEN']"
df_top4 = data['artist_name'] == "['Frank Ocean']"
df_top5 = data['artist_name'] == "['Honey Ivy']"

print(df_top2)


# In[243]:


# for energy
fig, ax = plt.subplots(figsize=(20,10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook",font_scale=2, rc={"lines.linewidth": 2})

sns.distplot(data['energy'], color='y',label="Эрнест Хемингуэй")
sns.distplot(data['energy'], color='b',label="Francisco Canaro")
sns.distplot(data['energy'], color='m',label="Эрих Мария Ремарк")
sns.distplot(data['energy'], color='g',label="Ignacio Corsini")
sns.distplot(data['energy'], color='r',label="Frank Sinatra")


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'for all the audio characteristics'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30, ha='left')
plt.rcParams["xtick.labelsize"] = 15


ax.set_title('energy DISTRIBUTION FROM DIFFERENT ARTISTS')
ax.legend(fontsize = 14)

# for valence
fig, ax = plt.subplots(figsize=(20,10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook",font_scale=2, rc={"lines.linewidth": 2})

sns.distplot(df_top1['valence'], color='y',label="Эрнест Хемингуэй")
sns.distplot(df_top2['valence'], color='b',label="Francisco Canaro")
sns.distplot(df_top3['valence'], color='m',label="Эрих Мария Ремарк")
sns.distplot(df_top4['valence'], color='g',label="Ignacio Corsini")
sns.distplot(df_top5['valence'], color='r',label="Frank Sinatra")


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'for all the audio characteristics'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30, ha='left')
plt.rcParams["xtick.labelsize"] = 15


ax.set_title('valence DISTRIBUTION FROM DIFFERENT ARTISTS')
ax.legend(fontsize = 14)

# danceability
fig, ax = plt.subplots(figsize=(20,10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook",font_scale=2, rc={"lines.linewidth": 2})

sns.distplot(df_top1['danceability'], color='y',label="Эрнест Хемингуэй")
sns.distplot(df_top2['danceability'], color='b',label="Francisco Canaro")
sns.distplot(df_top3['danceability'], color='m',label="Эрих Мария Ремарк")
sns.distplot(df_top4['danceability'], color='g',label="Ignacio Corsini")
sns.distplot(df_top5['danceability'], color='r',label="Frank Sinatra")


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'for all the audio characteristics'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30, ha='left')
plt.rcParams["xtick.labelsize"] = 15


ax.set_title('danceability DISTRIBUTION FROM DIFFERENT ARTISTS')
ax.legend(fontsize = 14)

# for liveness
fig, ax = plt.subplots(figsize=(20,10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook",font_scale=2, rc={"lines.linewidth": 2})

sns.distplot(data['liveness'], color='y',label="Эрнест Хемингуэй")
sns.distplot(df_top2['liveness'], color='b',label="Francisco Canaro")
sns.distplot(df_top3['liveness'], color='m',label="Эрих Мария Ремарк")
sns.distplot(df_top4['liveness'], color='g',label="Ignacio Corsini")
sns.distplot(df_top5['liveness'], color='r',label="Frank Sinatra")


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'for all the audio characteristics'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30, ha='left')
plt.rcParams["xtick.labelsize"] = 15


ax.set_title('liveness DISTRIBUTION FROM DIFFERENT ARTISTS')
ax.legend(fontsize = 14)

# for loudness
fig, ax = plt.subplots(figsize=(20,10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook",font_scale=2, rc={"lines.linewidth": 2})

sns.distplot(df_top1['loudness'], color='y',label="Эрнест Хемингуэй")
sns.distplot(df_top2['loudness'], color='b',label="Francisco Canaro")
sns.distplot(df_top3['loudness'], color='m',label="Эрих Мария Ремарк")
sns.distplot(df_top4['loudness'], color='g',label="Ignacio Corsini")
sns.distplot(df_top5['loudness'], color='r',label="Frank Sinatra")


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'for all the audio characteristics'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30, ha='left')
plt.rcParams["xtick.labelsize"] = 15


ax.set_title('loudness DISTRIBUTION FROM DIFFERENT ARTISTS')
ax.legend(fontsize = 14)

# fro tempo 
fig, ax = plt.subplots(figsize=(20,10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook",font_scale=2, rc={"lines.linewidth": 2})

sns.distplot(df_top1['tempo'], color='y',label="Эрнест Хемингуэй")
sns.distplot(df_top2['tempo'], color='b',label="Francisco Canaro")
sns.distplot(df_top3['tempo'], color='m',label="Эрих Мария Ремарк")
sns.distplot(df_top4['tempo'], color='g',label="Ignacio Corsini")
sns.distplot(df_top5['tempo'], color='r',label="Frank Sinatra")


labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = 'for all the audio characteristics'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30, ha='left')
plt.rcParams["xtick.labelsize"] = 15


ax.set_title('tempo DISTRIBUTION FROM DIFFERENT ARTISTS')
ax.legend(fontsize = 14)


# In[245]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('tableau-colorblind10')

sns.countplot(data['key'], palette = 'BuPu')
plt.title('Comparison of key', fontweight = 30, fontsize = 20)
plt.xlabel('key')
plt.ylabel('count')
plt.xticks(rotation = 90)
plt.show()


# In[250]:


from yellowbrick.features import JointPlotVisualizer


feature_names_art = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'valence','duration_ms']

X, y = data_x[feature_names], data['track_popularity']

features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize'] = (12,6)
visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()  

# Instantiate the visualizer
visualizer = JointPlotVisualizer(columns="danceability")

visualizer.fit_transform(X, y)        # Fit and transform the data
visualizer.show()  


# In[248]:


# Instantiate the visualizer
visualizer = JointPlotVisualizer(columns="data")

plt.rcParams['figure.figsize'] = (12,6)
visualizer.fit_transform(X, y)        # Fit and transform the data
visualizer.show() 


# In[251]:


##MODEL BUILDING 
use_col = ['acousticness','danceability','loudness','track_popularity','duration_ms','energy','speechiness','valence']


# In[253]:


cor = df.corr()
sns.heatmap(cor)


# In[257]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X = feat_cols.drop(columns=['track_popularity'])
y = feat_cols['track_popularity']


# In[255]:


x_train,x_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)
print("num of  train sample in train set:",x_train.shape)
print("Number of samples in validation set:",y_test.shape)


# In[256]:


from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()

random_forest.fit(x_train, y_train)
Y_pred_rf = random_forest.predict(x_test)
random_forest.score(x_train,y_train)
acc_random_forest = round(random_forest.score(x_train,y_train) * 100, 2)

print("Important features")
pd.Series(random_forest.feature_importances_,x_train.columns).sort_values(ascending=True).plot.barh(width=0.8)
print('__'*30)
print(acc_random_forest)


# In[ ]:




