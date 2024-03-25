import time 
import re
import pytz
import emoji
import datetime

def get_timestamp():
    "Store a timestamp for when training started."
    timestamp = time.time()
    timezone = pytz.timezone("Etc/GMT+7")
    dt = datetime.datetime.fromtimestamp(timestamp, timezone)
    return dt.strftime("%Y-%m-%d:%H:%m:%S")

def preprocess_text(text: str) -> str:  
    text = emoji.demojize(text)  
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

def flatten(z):
    return [x for y in z for x in y]
