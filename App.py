import altair as alt
import pandas as pd
import streamlit as st

# Load data
data = pd.read_csv('data/tracks_spotify_abhi.csv')

# Task 1: Sort dataframe by 'track_popularity' in descending order and select the top 20 tracks
top_20_tracks = data.sort_values(by='track_popularity', ascending=False).head(20)
st.write("Top 20 tracks by popularity:")
st.write(top_20_tracks[['track_name', 'track_popularity', 'key', 'danceability']])

# Task 2: Distribution of Major vs Minor Mode for random artist list
mode_counts = data['mode'].value_counts()
values = mode_counts.values
labels = ['Major', 'Minor']

# Create Altair pie chart
pie_chart = alt.Chart(pd.DataFrame({'values': values, 'labels': labels})).mark_bar().encode(
    x='labels',
    y='values',
    tooltip=['labels', 'values']
).properties(
    title='Distribution of Major vs Minor Mode'
).interactive()

# Task 3: Popularity by Key
box_chart = alt.Chart(data).mark_boxplot().encode(
    x='key',
    y='track_popularity',
    color='key:N',
    tooltip=['key', 'track_popularity']
).properties(
    title='Popularity by Key'
).interactive()

# Display charts using Streamlit
st.altair_chart(pie_chart, use_container_width=True)
st.altair_chart(box_chart, use_container_width=True)
