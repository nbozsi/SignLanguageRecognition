import subprocess
from glob import glob
import os
import json
from youtube_scraper import make_filepath


for folder, _, files in os.walk("videos"):
    print(folder)
    for i, file in enumerate(files):
        print("\t", i, file)
with open("MS-ASL/MSASL_train.json", "r") as f:
    records = json.load(f)
print(len(records))
print(len(glob("videos/**/*.mp4")))
input()

# trimming videos and setting fps to 24
counters = [0] * 1000
for row in records:
    if not os.path.exists(f'cropped/{row["clean_text"]}'):
        os.mkdir(f'cropped/{row["clean_text"]}')

    path = make_filepath(row)
    try:
        WIDTH, HEIGHT = row["width"], row["height"]
        X, Y = int(WIDTH * row["box"][1]), int(HEIGHT * row["box"][0])
        W, H = int(WIDTH * row["box"][3] - X), int(HEIGHT * row["box"][2] - Y)

        new_path = f"cropped/{row["clean_text"]}/{row["clean_text"]}_{counters[row["label"]]:03}.mp4"
        if os.path.exists(path):
            subprocess.run(["ffmpeg", "-i", path, "-vf", f"fps=24, crop={W}:{H}:{X}:{Y}", new_path])
            # os.remove(path)
            counters[row["label"]] += 1

        row["path"] = new_path
    except KeyboardInterrupt:
        with open("MSASL_train_with_paths.json", "w") as f:
            records = json.dump(records, f)
        raise KeyboardInterrupt
