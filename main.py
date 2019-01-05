import json
import os
import sqlite3
import sys
import warnings
from json.decoder import JSONDecodeError

import pandas as pd
import spotipy
import spotipy.util as util
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def delete_tracks():
    c.execute("""DELETE FROM user_tracks WHERE username LIKE ?""", (spotify_username,))


def get_user_playlists():
    global user_playlists
    offset = 0
    while True:
        playlists = spotify.current_user_playlists(limit=50, offset=offset)
        user_playlists += playlists['items']
        offset += 50
        if offset >= playlists['total']:
            break


def add_to_playlist():
    global recommend_df, user_playlists
    recommend_tracks = [row['track_id'] for index, row in recommend_df.iterrows() if row['isLiked'] == 1]
    if not user_playlists:
        get_user_playlists()
    for i in user_playlists:
        if i['name'] == 'r: ' + liked_playlist and i['owner']['id'] == spotify_username:
            playlist_id = i['id']
            break
    else:
        playlist = spotify.user_playlist_create(spotify_username, 'r: ' + liked_playlist, public=False)
        playlist_id = playlist['id']
    spotify.user_playlist_replace_tracks(spotify_username, playlist_id, recommend_tracks[:100])
    for i in range(100, len(recommend_tracks), 100):
        spotify.user_playlist_add_tracks(spotify_username, playlist_id, recommend_tracks[i:i + 100])


def get_tracks_from_database():
    global liked_artist_ids, all_tracks
    all_tracks.clear()
    related_artist_ids = liked_artist_ids[:]
    for i in range(0, len(liked_artist_ids), 100):
        query = """SELECT DISTINCT related_artist_id FROM related_artists WHERE artist_id IN ({i})""".format(
            i=','.join(['?'] * len(liked_artist_ids[i:i + 100])))
        c.execute(query, liked_artist_ids[i:i + 100])
        result = c.fetchall()
        related_artist_ids += [x[0] for x in result]
    for i in range(0, len(related_artist_ids), 100):
        query = """SELECT * FROM (SELECT * FROM song_information WHERE artist_id IN ({i}) ORDER BY RANDOM()) 
        GROUP BY artist_id ORDER BY RANDOM() LIMIT 20""".format(i=','.join(['?'] * len(related_artist_ids[i:i + 100])))
        c.execute(query, related_artist_ids[i:i + 100])
        all_tracks += c.fetchall()


def get_related_artists():
    update_token()
    c.execute("""SELECT DISTINCT artist_id,artist_name FROM user_tracks WHERE isLiked LIKE 1 AND username LIKE ? AND 
    artist_id NOT IN (SELECT DISTINCT artist_id FROM related_artists)""", (spotify_username,))
    artists = c.fetchall()
    if not artists:
        return
    insert_related = []
    for artist in artists:
        related_artists = spotify.artist_related_artists(artist[0])['artists']
        print(artist[1])
        insert_related += [(artist[1], artist[0], i['name'], i['id']) for i in related_artists]
        insert_related.append((artist[1], artist[0], artist[1], artist[0]))
    if insert_related:
        with connection:
            c.executemany("""INSERT INTO related_artists VALUES (?,?,?,?)""", insert_related)


def get_song_information():
    global all_tracks
    update_token()
    insert_tracks = []
    features = []
    track_ids = [x[0] for x in all_tracks]
    for i in range(0, len(all_tracks), 50):
        features += spotify.audio_features(track_ids[i:i + 50])
    for i in range(len(all_tracks)):
        t = all_tracks[i]
        f = features[i]
        if not f:
            continue
        print('Song information:', t[3])
        insert_tracks.append((t[0], t[1], t[2], t[3], f['acousticness'], f['danceability'], f['energy'],
                              f['duration_ms'], f['instrumentalness'], f['key'], f['liveness'], f['loudness'],
                              f['mode'], f['speechiness'], f['tempo'], f['time_signature'], f['valence']))
    if insert_tracks:
        with connection:
            c.executemany("""INSERT INTO song_information VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", insert_tracks)
        insert_tracks.clear()


def get_tracks(album_ids):
    global all_tracks
    albums = []
    for i in range(0, len(album_ids), 20):
        albums += spotify.albums(album_ids[i:i + 20])['albums']
    for album in albums:
        all_tracks += [(t['id'], t['name'], album['artists'][0]['id'], album['artists'][0]['name']) for t in
                       album['tracks']['items']]


def get_albums(artist_id, artist_name):
    update_token()
    c.execute("""SELECT artist_id FROM song_information WHERE artist_id LIKE ?""", (artist_id,))
    artist_ids = c.fetchall()
    if artist_ids:
        return True
    types = [["album", 50], ["single", 20], ["compilation", 10]]
    album_ids = []
    for album_type in types:
        albums = spotify.artist_albums(artist_id=artist_id, album_type=album_type[0], limit=album_type[1])
        album_ids += [album['id'] for album in albums['items']]
    if album_ids:
        get_tracks(album_ids)
    else:
        with connection:
            c.execute("""INSERT INTO excluded_artists VALUES (?,?)""", (artist_name, artist_id))


def get_artists():
    c.execute("""SELECT DISTINCT artist_id,artist_name FROM related_artists WHERE 
    artist_id NOT IN (SELECT artist_id FROM excluded_artists) AND
    artist_id NOT IN (SELECT DISTINCT artist_id FROM song_information)""")
    artist_ids = c.fetchall()
    if not artist_ids:
        return
    for i in artist_ids:
        print('Getting:', i[1])
        get_albums(i[0], i[1])


def get_user_song_information(tracks, is_liked):
    global user_tracks, liked_artist_ids
    track_ids = [i['id'] for i in tracks]
    insert_tracks = []
    features = []
    for i in range(0, len(track_ids), 50):
        features += spotify.audio_features(track_ids[i:i + 50])
    for i in range(len(tracks)):
        t = tracks[i]
        f = features[i]
        if not f:
            continue
        if is_liked and t['artists'][0]['id'] not in liked_artist_ids:
            liked_artist_ids.append(t['artists'][0]['id'])
        insert_tracks.append((t['id'], t['name'], t['artists'][0]['id'], t['artists'][0]['name'], spotify_username,
                              f['acousticness'], f['danceability'], f['energy'], f['duration_ms'],
                              f['instrumentalness'], f['key'], f['liveness'], f['loudness'], f['mode'],
                              f['speechiness'], f['tempo'], f['time_signature'], f['valence'], is_liked))
    if insert_tracks:
        user_tracks += insert_tracks
        with connection:
            c.executemany("""INSERT INTO user_tracks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", insert_tracks)
        insert_tracks.clear()


def get_playlist_tracks(playlist_name, is_liked):
    global user_playlists
    if not user_playlists:
        get_user_playlists()
    for playlist in user_playlists:
        if playlist['name'].lower() == playlist_name.lower():
            playlist_id = playlist['id']
            break
    else:
        raise Exception('Playlist not found: ' + playlist_name)
    playlist_tracks = []
    offset = 0
    while True:
        tracks = spotify.user_playlist_tracks(spotify_username, playlist_id, limit=100, offset=offset)
        playlist_tracks += [i['track'] for i in tracks['items']]
        offset += 100
        if offset >= tracks['total'] or offset >= 200:
            break
    get_user_song_information(playlist_tracks, is_liked)


def get_top_tracks():
    long_term = spotify.current_user_top_tracks(limit=50, time_range="long_term")
    medium_term = spotify.current_user_top_tracks(limit=50, time_range="medium_term")
    short_term = spotify.current_user_top_tracks(limit=50, time_range="short_term")
    top_tracks = long_term['items'] + medium_term['items'] + short_term['items']
    temp = {json.dumps(d, sort_keys=True) for d in top_tracks}
    top_tracks = [json.loads(t) for t in temp]
    get_user_song_information(top_tracks, 1)


def get_user_tracks():
    global user_tracks, liked_artist_ids, ready
    c.execute("""SELECT * FROM user_tracks WHERE username LIKE ?""", (spotify_username,))
    user_tracks = c.fetchall()
    if not user_tracks:
        if liked_playlist == 'top tracks':
            get_top_tracks()
        else:
            get_playlist_tracks(liked_playlist, 1)
        get_playlist_tracks(disliked_playlist, 0)
    else:
        liked_artist_ids = [i[2] for i in user_tracks if i[-1] == 1]
        ready = True


# update token regularly to avoid session expire
def update_token():
    global spotify, token
    try:
        token = util.prompt_for_user_token(spotify_username, scope, client_id, client_secret, redirect_uri)
    except(AttributeError, JSONDecodeError):
        os.remove(f".cache-{spotify_username}")
        token = util.prompt_for_user_token(spotify_username, scope, client_id, client_secret, redirect_uri)
    spotify = spotipy.Spotify(auth=token)


# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# set Spotify API
client_id = os.environ.get("client_id")
client_secret = os.environ.get("client_secret")
redirect_uri = os.environ.get("redirect_uri")
spotify_username = os.environ.get("spotify_username")
scope = os.environ.get("scope")

token = None
spotify = spotipy.Spotify
update_token()

# create database connection
connection = sqlite3.connect('spotify.sqlite3')
c = connection.cursor()

all_tracks = []
liked_artist_ids = []
user_tracks = []
user_playlists = []
ready = False
liked_playlist = ''
disliked_playlist = 'disliked'

if len(sys.argv) > 1:
    if sys.argv[1] == '-d':
        liked_playlist = ' '.join(sys.argv[2:])
        delete_tracks()
        with connection:
            c.execute("""DELETE FROM last_playlist WHERE username LIKE ?""", (spotify_username,))
            c.execute("""INSERT INTO last_playlist VALUES (?,?)""", (spotify_username, liked_playlist))
else:
    c.execute("""SELECT playlist_name FROM last_playlist WHERE username LIKE ?""", (spotify_username,))
    result = c.fetchone()
    if result:
        liked_playlist = result[0]
    else:
        raise Exception("No playlist provided.\nUsage: python main.py -d playlist name")

get_user_tracks()
if not ready:
    get_related_artists()
    get_artists()
    if all_tracks:
        get_song_information()
get_tracks_from_database()

columns_user = ["track_id", "track_name", "artist_id", "artist_name", "username", "acousticness", "danceability",
                "energy", "duration_ms", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness",
                "tempo", "time_signature", "valence", "isLiked"]
columns_all = ["track_id", "track_name", "artist_id", "artist_name", "acousticness", "danceability", "energy",
               "duration_ms", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo",
               "time_signature", "valence"]
train_df = pd.DataFrame.from_records(user_tracks, columns=columns_user, index='track_id')
test_df = pd.DataFrame.from_records(all_tracks, columns=columns_all, index='track_id')

X = train_df.loc[:, 'acousticness':'valence'].as_matrix().astype('float')
y = train_df['isLiked'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
test_X = test_df.loc[:, 'acousticness':].as_matrix().astype('float')

rfc = RandomForestClassifier(n_estimators=200, max_features='log2', min_samples_leaf=1)
rfc.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

rfc_accuracy = accuracy_score(y_test, rfc.predict(X_test))
gnb_accuracy = accuracy_score(y_test, gnb.predict(X_test))

if rfc_accuracy >= gnb_accuracy:
    predictions = rfc.predict(test_X)
    print("accuracy_score:", accuracy_score(y_test, rfc.predict(X_test)))
    print("confusion_matrix:\n", confusion_matrix(y_test, rfc.predict(X_test)))
    print("precision_score:", precision_score(y_test, rfc.predict(X_test)))
    print("recall_score:", recall_score(y_test, rfc.predict(X_test)))
else:
    predictions = gnb.predict(test_X)
    print("accuracy_score:", accuracy_score(y_test, gnb.predict(X_test)))
    print("confusion_matrix:\n", confusion_matrix(y_test, gnb.predict(X_test)))
    print("precision_score:", precision_score(y_test, gnb.predict(X_test)))
    print("recall_score:", recall_score(y_test, gnb.predict(X_test)))

recommend_df = pd.DataFrame({'track_id': test_df.index, 'isLiked': predictions})

add_to_playlist()
