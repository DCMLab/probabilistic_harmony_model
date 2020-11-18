import pandas as pd
import numpy as np
#import mozart_piano_sonatas.utils.feature_matrices as fm
from utils import notetype, load_dcml_tsv, name2tpc
import glob
import os.path as path
import tqdm

# predefined values
# -----------------

# theoretical chord tones of the chord types in the corpus
# (pitch class is given in fifths)
chord_types = {
    'M': [0,1,4],
    'm': [0,1,-3],
    'o': [0,-3,-6],
    '+': [0,4,8],
    'Mm7': [0,1,4,-2],
    'mm7': [0,1,-3,-2],
    'MM7': [0,1,4,5],
    'mM7': [0,1,-3,5],
    '%7': [0,-3,-6,-2],
    'o7': [0,-3,-6,-9],
    '+7': [0,4,8,-2]
}

# corpus directory

# helper function
# ---------------

def total_onsets(events, measures):
    """
    Returns the onsets of events (chord labels or notes) relative to the beginning of the piece
    by converting them from measure-relative notation
    """
    moffsets = measures.act_dur.values.cumsum()
    monsets = moffsets - measures.act_dur.values
    mi = events.mc - 1
    return events.mc_onset + monsets[mi]

def merge_ties(notes):
    """
    Returns a copy of notes with ties merged.
    """
    notes = notes.copy()
    beginnings = notes[notes.tied == 1]
    continues = notes[notes.tied < 1]
    for i in beginnings.index:
        on = notes.total_onset[i]
        off = notes.total_offset[i]
        midi = notes.midi[i]
        tpc = notes.tpc[i]
        while True:
            cont = continues[(continues.total_onset == off) &
                             (continues.midi == midi) &
                             (continues.tpc == tpc)].first_valid_index()
            if cont is None:
                break
            off = continues.total_offset[cont]
            if continues.tied[cont] == -1:
                break
        notes.at[i, 'total_offset'] = off
        notes.at[i, 'duration'] = off - on
    return notes[~(notes.tied < 1).fillna(False)]

def load_dfs(corpus, piece):
    """
    Loads and preprocesses dataframes for the notes and chord labels of a piece.
    """
    measures = load_dcml_tsv(corpus, piece, 'measures')
    
    notes = load_dcml_tsv(corpus, piece, 'notes')
    notes['total_onset'] = total_onsets(notes, measures)
    notes['total_offset'] = notes.total_onset.values + notes.duration.values
    notes = merge_ties(notes)
    max_offset = notes.total_offset.values.max()
    
    harmonies = load_dcml_tsv(corpus, piece, 'harmonies')
    harmonies = harmonies[~harmonies.chord.isnull()]
    harmonies['total_onset'] = total_onsets(harmonies, measures)
    harmonies['total_offset'] = np.append(harmonies.total_onset.values[1:], max_offset)
    
    return notes, harmonies

# extracting chords
# -----------------

def get_chords(notes, harmonies, id_offset=0):
    """
    Computes chords as label x note pairs for a piece (given by its notes and chord labels).
    Pairs that belong to the same chord get the same id, starting from id_offset.
    Returns the dataframe of chords and the highest used id.
    """
    key = name2tpc(harmonies.globalkey[0])

    # setup the columns of the result dataframe
    chordids = np.empty(0, dtype=int)
    labels   = np.empty(0, dtype=str)
    fifths   = np.empty(0, dtype=int)
    types    = np.empty(0, dtype=str)
    # running id counter
    highest_id = id_offset

    # for checking whether the chord label is empty at some point
    chord_is_null = harmonies.chord.isnull()

    # iterate over all harmonies
    for i, ih in enumerate(harmonies.index):
        # chord label empty or '@none'? then skip
        if chord_is_null[ih] or harmonies.chord[ih] == '@none':
            continue

        # get info about the current harmony
        on = harmonies.total_onset[ih]
        off = harmonies.total_offset[ih]
        label = harmonies.chord_type[ih]
        root = harmonies.root[ih] + key

        # compute the corresponding notes, their pitches, and their note types
        inotes = (notes.total_offset > on) & (notes.total_onset < off)
        pitches = notes.tpc[inotes].values - root
        chord_tones = chord_types[label]
        note_types = [notetype(p, pitches, chord_tones) for p in pitches]

        # add everything to the dataframe columns
        chordids = np.append(chordids, np.repeat(i + id_offset, len(pitches)))
        labels   = np.append(labels, np.repeat(label, len(pitches)))
        fifths   = np.append(fifths, pitches)
        types    = np.append(types, note_types)
        highest_id = i + id_offset

    # create the result dataframe
    chords_df = pd.DataFrame({
        'chordid': chordids,
        'label': labels,
        'fifth': fifths,
        'type': types})
    return chords_df, highest_id

# processing files
# ----------------

def get_chords_from_piece(folder, file, id_offset=0):
    """
    Same as get_chords, but takes a corpus subdirectory and a piece id
    """
    notes, harmonies = load_dfs(folder, file)
    return get_chords(notes, harmonies, id_offset)

def get_chords_from_files(filelist):
    """
    Returns the combined chords for several pieces.
    Takes a list of subdirectory x piece pairs.
    """
    files = []
    offset = 0
    all_chords = None
    for folder, file in tqdm.tqdm(filelist):
        try:
            chords, max_id = get_chords_from_piece(folder, file, offset)
            all_chords = chords if (all_chords is None) else all_chords.append(chords)
            offset = max_id + 1
            files.append(f"{folder} {file}");
        except FileNotFoundError:
            print(f'file not found for {folder} {file}')
            continue
        except ValueError:
            print(f'ValueError in {folder} {file}')
            continue
        except (KeyboardInterrupt):
            print("interrupted by user, exiting.")
            quit()
        except:
            print(f'error while processing {folder} {file}')
            #raise Exception(f"failed file: {folder} {file}")
    print(f"got {max_id} chords and {len(all_chords)} notes from the {len(files)} files listed in preprocess_dcml.log")
    with open("preprocess_dcml.log","w") as f:
      print(f"got {max_id} chords and {len(all_chords)} notes from the following {len(files)} files",file=f)
      f.write("\n".join(files))

    return all_chords.reset_index(drop=True)

def get_corpus_pieces(corpus):
    """
    Returns a list of pieces in a corpus as subdirectory x piece pairs.
    """
    dirs = glob.glob(path.join(corpus, 'annotations', '*'))
    files = [(d, path.splitext(path.basename(f))[0])
             for d in dirs
             for f in glob.glob(path.join(d, 'notes', '*.tsv'))]
    return files

# script
# ------

if __name__ == "__main__":
    print("scanning corpus...")
    pieces = get_corpus_pieces(path.join("data", "dcml_corpora"))
    print("extracting chords from pieces...")
    all_chords = get_chords_from_files(pieces)
    print("writing chords...")
    all_chords.to_csv(path.join('data', 'dcml.tsv'), sep='\t', index=False)
    print("done.")
