# import music21 as m21
# import glob
import tqdm
from utils import notetype
from pathlib import Path
# import muspy
import multiprocessing as mp

import preprocess_wikifonia as prep

def read_file(filename):
    try:
        chords, n, uk = prep.readchords(filename, iszip=True)
        return (chords, n, uk, filename)
    except Exception as e:
        print(f"Could not read chords from {filename}:\n{e}")
        return None

if __name__ == "__main__":
    allchords = []
    processed_files = []
    nnotes = 0
    unknown_types = set()

    print("extracting chords from pieces...")
    all_files = list(Path("data", "ewld", "dataset").glob("**/*.mxl"))
    print(f"found {len(all_files)} files.")
    with mp.Pool(3) as pool:
        outputs = list(tqdm.tqdm(pool.imap(read_file, all_files), total = len(all_files)))
    print("collecting outputs")
    for output in tqdm.tqdm(outputs):
        if output is not None:
            chords, n, uk, filename = output # prep.readchords(file, iszip=True)
            allchords += chords
            processed_files.append(filename)
            nnotes += n
            unknown_types = unknown_types.union(uk)
        # except Exception as e:
        #     print(f"Could not read chords from {file}:\n{e}")

    with open(Path("data", "preprocess_ewld.txt"),"w") as f:
      print(f"got {len(allchords)} chords and {nnotes} notes from the following {len(processed_files)} files:",file=f)
      f.write("\n".join(map(str, processed_files)))
    print(f"got {len(allchords)} chords and {nnotes} notes from the {len(processed_files)} files listed in data/preprocess_ewld.txt")
    print("The following chord types could not be interpreted and are ignored:", unknown_types)
    print("writing chords...")
    prep.writechords(Path("data", "ewld.tsv"), allchords)
    print("done.")
