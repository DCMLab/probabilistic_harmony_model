import music21 as m21
import glob
import tqdm
from utils import notetype
from os import path, makedirs
import muspy

import preprocess_wikifonia as prep

if __name__ == "__main__":
    # ensure dataset is present
    # print("downloading data...")
    # makedirs(path.join("data","wikifonia"),exist_ok=True)
    # muspy.WikifoniaDataset(path.join("data", "wikifonia"), download_and_extract=True)
    allchords = []
    files = []
    nnotes = 0
    unknown_types = set()

    print("extracting chords from pieces...")
    for file in tqdm.tqdm(glob.glob(path.join("data", "ewld", "unknown", "*.xml"))):
        try:
            chords, n, uk = prep.readchords(file, iszip=False)
            allchords += chords
            files.append(file)
            nnotes += n
            unknown_types = unknown_types.union(uk)
        except Exception as e:
            print(f"Could not read chords from {file}:\n{e}")

    with open(path.join("data", "preprocess_ewld.log"),"w") as f:
      print(f"got {len(allchords)} chords and {nnotes} notes from the following {len(files)} files:",file=f)
      f.write("\n".join(files))
    print(f"got {len(allchords)} chords and {nnotes} notes from the {len(files)} files listed in data/preprocess_ewld.log")
    print("The following chord types could not be interpreted and are ignored:", unknown_types)
    print("writing chords...")
    prep.writechords(path.join("data", "ewld.tsv"), allchords)
    print("done.")
