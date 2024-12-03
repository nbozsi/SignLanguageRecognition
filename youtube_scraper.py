import os
from glob import glob
import yt_dlp
from yt_dlp.utils import download_range_func

import json
from pprint import pprint

make_filepath = lambda x: f"videos/{x["clean_text"]}/{x["clean_text"]}_{x['url'][-11:]}.mp4"


def main():
    video_paths = set(glob("videos/**/*.mp4"))
    with open("MS-ASL/MSASL_train.json", "r") as f:
        records = json.load(f)

    print(f"{len(records)} records found")
    records = list(filter(lambda x: make_filepath(x) not in video_paths, records))
    print(f"{len(records)} left for download")

    ydl_opts = {
        "format": "best",
        "download_ranges": None,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",  # Ensures format compatibility
                "preferedformat": "mp4",  # Specify output format if needed
            },
        ],
    }
    private_videos = 0
    input("Press Enter to start!")
    for row in records[::-1]:
        print(row)

        # naming
        if not os.path.exists(f"videos/{row["clean_text"]}"):
            os.mkdir(f"videos/{row["clean_text"]}")

        file_name = make_filepath(row)

        # updating ydl options
        ydl_opts["outtmpl"] = file_name
        ydl_opts["download_ranges"] = download_range_func(None, [(row["start_time"], row["end_time"])])

        # downloading
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([row["url"]])
        except yt_dlp.utils.DownloadError:
            private_videos += 1
        except KeyboardInterrupt:
            with open("MSASL_train_with_paths.json", "w") as f:
                records = json.dump(records, f)
            raise KeyboardInterrupt
        # updating database with the path
        row["path"] = file_name

    print(f"{private_videos = }")

    with open("MSASL_train_with_paths.json", "w") as f:
        records = json.dump(records, f)


if __name__ == "__main__":
    main()
