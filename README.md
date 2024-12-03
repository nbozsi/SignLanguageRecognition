# Sign Language Recognition
Advanced Machine Learning Course BME 2024 Fall
## Goal
To build a neural network that recognizes which words are signed in a video.

## The Data
We used the MS-ASL American Sign Language Dataset dataset, it's available [here](https://www.microsoft.com/en-us/download/details.aspx?id=100121).
Target Variable:
  - clean_text: the word
Features:
  - url: youtube link of the video
  - box: bounding box of the signer in the video
  - start_time: timestamp where the word starts in the video
  - end_time: timestamp where the word ends in the video
  - other metadata about the video

Unfortunately, only ~10k of the 17k+ videos were available, that's ~9.5 videos for each distinct word.
