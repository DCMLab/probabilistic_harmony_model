import music21 as m21
import glob
import tqdm
from utils import notetype
from os import path, makedirs
import muspy

# chord types
# -----------

# maps MusicXML chord kinds to chord tones in fifths
# e.g. [0,1,4] = [P1, P5, M3]
CHORD_TYPES = {
    # triads
    "major": [0,1,4],
    "minor": [0,1,-3],
    "augmented": [0,4,8],
    "diminished": [0,-3,-6],
    # sevenths
    "dominant": [0,1,4,-2],
    "major-seventh": [0,1,4,5],
    "minor-seventh": [0,1,-3,-2],
    "diminished-seventh": [0,-3,-6,-9], # full-diminished
    "augmented-seventh": [0,4,8,-2],
    "half-diminished": [0,-3,-6,-2],
    "major-minor": [0,1,-3,5],
    # sixths
    "major-sixth": [0,1,4,3],
    "minor-sixth": [0,1,-3,3], # it's a bit unclear if M6 or m6 is meant (probably M6)
    # ninths
    "dominant-ninth": [0,1,4,-2,2],
    "major-ninth": [0,1,4,5,2],
    "minor-ninth": [0,1,-3,-2,2],
    # 11ths
    "dominant-11th": [0,1,4,-2,2,-1],
    "major-11th": [0,1,4,5,2,-1],
    "minor-11th": [0,1,-3,-2,2,-1],
    # 13ths
    "dominant-13th": [0,1,4,-2,2,-1,3],
    "major-13th": [0,1,4,5,2,-1,3],
    "minor-13th": [0,1,-3,-2,2,-1,3],
    # suspended / other
    "suspended-second": [0,1,2],
    "suspended-fourth": [0,1,-1], # should this just point to dominant? 4th is probably an ornament.
    "power" : [0,1]
    # rest does not occur or is ignored
}

CHORD_ALT_TYPES = {
    "min": "minor",
    "aug": "augmented",
    "sus47": "dominant", # suspension is assumed to be an ornament here
    "7sus": "dominant",
    "dim7": "half-diminished",
    "dim": "diminished",
    "dominant-seventh": "dominant",
    "minor-major": "major-minor", # presumably
    "minMaj7": "major-minor",
    "min6": "minor-sixth",
    "9": "dominant-ninth",
    "6": "major-sixth",
    "7": "dominant",
    "maj7": "major-seventh",
    "min7": "minor-seventh",
    "maj9": "major-ninth",
    "min9": "minor-ninth",
    "maj69": "major-13th",
    "m7b5": "half-diminished",
}

# ignored chord types
# - pedal
# - min/G
# - none
# - /A
# - other
# - augmented-ninth
# - '' (empty)

# helper functions
# ----------------

def getonset(elt):
    """
    Returns the absolute onset of an element.
    """
    return elt.offset

def getoffset(elt):
    """
    Returns the absolute offset of an element.
    """
    return elt.offset + elt.duration.quarterLength

def spellpc(pitch):
    """
    Takes a music21 pitch and returns its tonal pitch class on the line of 5ths (C=0, G=1, F=-1 etc.)
    """
    diastemtone = (pitch.diatonicNoteNum - 1 ) % 7
    diafifths = (diastemtone * 2 + 1) % 7 - 1
    fifths = diafifths + int(pitch.alter) * 7
    return fifths

def getchordtones(chordtype):
    """
    Returns the chord tones that belong to each label
    """
    return CHORD_TYPES[chordtype]

def getchordtype(label):
    """
    Returns the normalized chord type for a given label.
    If the type given in the label is no known, None is returned.
    """
    kind = label.chordKind
    if kind in CHORD_TYPES:
        return kind
    elif kind in CHORD_ALT_TYPES:
        return CHORD_ALT_TYPES[kind]
    else:
        return None

# extracting chords
# -----------------

def getannotations(piece):
    """
    Takes a music21 piece and returns a list of triples
    (label, onset, offset) for each chord annotation in the piece.
    """

    # get all chord labels
    piece = piece.flat
    labels = list(piece.getElementsByClass(m21.harmony.ChordSymbol))

    # compute the offsets (= onset of the next chord or end of piece)
    endofpiece = piece.highestOffset + 1.0 # piece.duration doesn't work
    offsets = [getonset(label) for label in labels][1:]
    offsets.append(endofpiece)

    # combine labels with onsets and offsets
    harmonies = [(label, getonset(label), offset) for (label, offset) in zip(labels, offsets)]
    return harmonies

def getchords(piece):    
    """
    Takes a music21 piece and returns a list of chords
    (label plus notes with note types).
    """ 
    
    harm = getannotations(piece) # parse chord symbols, calls function getharmonies
    harmonies = [] # start from empty list
    nnotes = 0
    unknown_types = set()
    
    for label, chord_onset, chord_offset in harm: # iterate over all elements of harm 

        chordtype = getchordtype(label)

        if chordtype == None:
            unknown_types.add(label.chordKind)
            continue
        
        if label.root() == None:
            continue

        root = spellpc(label.root())
        chordtones = getchordtones(chordtype)    
        
        notes = piece.flat.getElementsByClass(m21.note.Note)
        pitches = [spellpc(note.pitch) for note in notes
                   if getonset(note) < chord_offset and getoffset(note) > chord_onset] 
            
        notes = [(pitch-root, notetype(pitch, pitches, chordtones)) for pitch in pitches]
        nnotes += len(notes)
             
        harmonies.append((chordtype, notes))
        
    return harmonies, nnotes, unknown_types

def readfile(filename):
    """
    Reads filename as a MusicXML file, returns a music21 stream.
    """
    am = m21.converter.ArchiveManager(filename, archiveType='zip')
    return m21.converter.parse(am.getData(), format="musicxml")

def readchords(filename):
    """
    Reads a MusicXML file and returns the contained chords.
    """
    return getchords(readfile(filename))

# saving chords
# -------------

def writechords(filename, chords):
    """
    Writes a list of chords as a TSV file.
    """
    with open(filename, 'w') as f:
        f.write("chordid\tlabel\tfifth\ttype\n")
        id = 0
        for chord in chords:
            for note in chord[1]:
                f.write("\t".join(map(str,[id,chord[0],note[0],note[1]])) + "\n")
            id += 1

# script
# ------

if __name__ == "__main__":
    # ensure dataset is present
    print("downloading data...")
    makedirs(path.join("data","wikifonia"),exist_ok=True)
    muspy.WikifoniaDataset(path.join("data", "wikifonia"), download_and_extract=True)
    allchords = []
    files = []
    nnotes = 0
    unknown_types = set()

    print("extracting chords from pieces...")
    for file in tqdm.tqdm(glob.glob(path.join("data", "wikifonia", "Wikifonia", "*.mxl*"))):
        try:
            chords, n, uk = readchords(file)
            allchords += chords
            files.append(file)
            nnotes += n
            unknown_types = unknown_types.union(uk)
        except Exception as e:
            print(f"Could not read chords from {file}:\n{e}")

    with open("preprocess_wikifonia.log","w") as f:
      print(f"got {len(allchords)} chords and {nnotes} notes from the following {len(files)} files:",file=f)
      f.write("\n".join(files))
    print(f"got {len(allchords)} chords and {nnotes} notes from the {len(files)} files listed in preprocess_wikifonia.log")
    print("The following chord types could not be interpreted and are ignored:", unknown_types)
    print("writing chords...")
    writechords(path.join("data", "wikifonia.tsv"), allchords)
    print("done.")
