import ee
from google.oauth2.credentials import Credentials

refresh_token = "<REFRESH_TOKEN>"
client_id = "<CLIENT_ID>"
client_secret = "<CLIENT_SECRET>"

creds = Credentials(
    token=None,
    refresh_token=refresh_token,
    token_uri="https://oauth2.googleapis.com/token",
    client_id=client_id,
    client_secret=client_secret
)

ee.Initialize(credentials=creds)
print("✅ GEE Login berhasil")
