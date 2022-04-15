import re
import os
import pandas as pd
import numpy as np
import youtube_dl
import urllib.request



# construct a URI like  URL data.yt8m.org/AB/ABCD.js
# map the pesudo random id to the real youtub id
def get_real_id(pesudo_id):
    url = "http://data.yt8m.org/2/j/i/{}/{}.js".format(pesudo_id[0:2], pesudo_id)
    response = urllib.request.urlopen(url).read().decode()
    real_id = response.split(",")[-1][1:-3]
    return real_id


### This function extract read youtube id and metadata based on the pusedo id
def get_video_metadata(pesudo_id):
    try:
        real_id  = get_real_id(pesudo_id)
        url = "https://www.youtube.com/watch?v=" + real_id
        ydl = youtube_dl.YoutubeDL()
        result = ydl.extract_info(url, download=False)
        fields = [
            "title",
            "categories",
            "tags",
            "description",
            "is_live",
            "view_count",
            "like_count",
            "channel_url",
            "duration",
            "average_rating",
            "age_limit",
            "webpage_url",
        ]
        video_metadata = [pesudo_id,real_id]
        video_metadata.append([result[field] for field in fields])
        return video_metadata
    except:
        return [np.nan for i in range(14)]



