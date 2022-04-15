# YouTube-8m-Classification
IDS705 Final Project: Erika Fox, Shufan Xia, Marlyne Hakizimana

This project contribute to progress in the video classification space by using the YouTube-8M dataset. We trained three different models with both detailed frame-level data and more summarized video-level data, and then tested them on three different styles of feature inputs (visual only, audio only, and combined audio/visual) in order to determine the optimal way to accurately assign multiple “tag” labels to videos. Our results provide useful insights as to what features are more important for video classification, and this work could be used to improve the experiences for both the content creators and audiences of YouTube. Our full report can be read in the file: `YouTubeClassificationReport.pdf` 


### Data:
  To get data, follow the instruction at [Youtube8M](https://research.google.com/youtube8m/download.html). Installation of `curl` is required.
  
  Create seprate folders for train, validation and test data, and download the  tfrecords. To download 1/100-th of the training data from the US use:<br>
  `curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python`. 
  
  To download 1/100-th of the validation data use:<br>
  `curl data.yt8m.org/download.py | partition=2/frame/validate mirror=us python`.
  
  For test data: the original  [Youtube8M](https://research.google.com/youtube8m/download.html) test data is unlabeled. We used a few tfrecords from the downloaded validation data, and deleted those from the validation folder.
  
### Dependencies:
  see `requirements.txt`


### User Instructions

1) Clone repo: `git clone https://github.com/erikanfox/YouTube-8m-Classification.git`
2) Create a pip/conda virtual environment
3) Install required packages: `pip install -r requirements.txt`
4) Download data (see information above)
5) Preprocess/divide the data using the files found the `preprocessing` folder:

    `preprocess.py` prepares the data for modeling
    This file is required for EDA and modeling
    
6) Exploratory data analysis uses the files found the folder `EDA`

    `get_metadata.py` fetches video meta data based on the pseudo_id from the originla data
    `EDA.ipynb` displays an exploratory data analysis,
    
7) Modeling: 
- Run models use the files found in the `modeling` folder:
      `video-level.ipynb` runs our video-level models,
      `frame-level.ipynb` runs our frame-level models,
      These notebooks require funcitons from `preprocess.py` from the `preprocessing` folder to prerpocess data and  `report.py` under the `metric` folder to monitor model performance and export prediction report for results analysis.
 - Model performance use the files found in the `metrics` folder:
     -   `average_precision_calculator.py`, `eval_util.py`, and `mean_average_precision_calculator.py` are borrowed from (youtube-8m Github Repo)[https://github.com/google/youtube-8m] 
     -  `report.py` uses these python scripts above to calculate gAP, Hit@1, PERR ([BenchMark paper](https://ui.adsabs.harvard.edu/abs/2016arXiv160908675A/abstract)). `report.py` also compute F-1 score and compile prediction results to a dataframe.
report.py
     - prediction on test data: for each of the three models and three combinations of inputs, test result dataframe is exported into the `data` folder.
      
8) Analyze results using the files found in the `results` folder:

     `metrics.ipynb` generates ROC and precision-recall plots based on the results in the `data` folder.

