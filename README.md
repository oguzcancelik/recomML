# Song Recommendation

- It's a machine learning project about recommending songs to a user based on user's music taste.

- **Note**: You need a Spotify API key to run this project properly. You can get one from [Spotify's official developer website](https://developer.spotify.com/). If you don't have a chance to get one, I can provide my own key to you.

- You need to fill **.env** file with your API information.
    - spotify_username="***"
    - client_id="***"
    - client_secret="***"
    - redirect_uri="***"
    
- All datas are stored in the **spotify.sqlite3** file. 
- All necessary Python libraries are stored in the **requirements.txt** file. To install these libraries first create a virtual environment, activate it and run the following command.
```sh
    pip install -r requirements.txt
``` 
- Usage: Get recommendation based on top tracks.
```sh
    python main.py -d top tracks
``` 
- Usage: Get recommendation based on a playlist.
```sh
    python main.py -d playlist name
``` 
- Now run **main.py**. First it's going to ask you to give permissions. After giving permissions, you will be redirected to an URL. Copy that URL and paste it to console.
```sh
 $  python main.py -d top tracks


            User authentication requires interaction with your
            web browser. Once you enter your credentials and
            give authorization, you will be redirected to
            a url.  Paste that url you were directed to to
            complete the authorization.


Opened https://accounts.spotify.com/authorize?client_id=fc9383&response_type=code&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2F&scope=playlist-modify-public+user-top-read in your browser


Enter the URL you were redirected to:


```
- After that user and song information will be get from the Spotify and stored in the database.

- Getting tracks of a playlist at the first time would take a few minutes. Running second time will be a lot faster.

- **Note:** Authentication only needed in the first time. 

- **Note:** You can ask any unclear point.