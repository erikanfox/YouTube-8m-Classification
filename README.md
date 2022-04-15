# YouTube-8m-Classification
IDS705 Final Project: Erika Fox, Shufan Xia, Marlyne Hakizimana

This project contribute to progress in the video classification space by using the YouTube-8M dataset. We trained three different models with both detailed frame-level data and more summarized video-level data, and then tested them on three different styles of feature inputs (visual only, audio only, and combined audio/visual) in order to determine the optimal way to accurately assign multiple “tag” labels to videos. Our results provide useful insights as to what features are more important for video classification, and this work could be used to improve the experiences for both the content creators and audiences of YouTube.


### Data:
  To get data, follow the instruction at [Youtube8M](https://research.google.com/youtube8m/download.html). 
  
  In 
  `curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python`
  
### Dependencies:
  see `requirements.txt`


### User Instructions

1) Clone repo: `git clone https://github.com/erikanfox/YouTube-8m-Classification.git`
2) Create a pip/conda virtual environment
3) Install required packages: `pip install -r requirements.txt`
4) Download data (see information above)
5) Preprocess/divide the data using the following files:


6) Run models using the folling files:
7) Analyze results using the following files: `metrics.ipynb`, `test_results.ipynb`
