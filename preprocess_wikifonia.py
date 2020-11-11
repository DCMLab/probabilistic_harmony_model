import music21 as m21
import glob
import tqdm
from utils import notetype
import os.path as path
import muspy

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

def getchordtones(label):
    """
    Returns the chord tones that belong to each label
    """
    return [spellpc(tone.pitch) for tone in label]

def getchordtype(label):
    return label.chordKind

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
    for label, chord_onset, chord_offset in harm: # iterate over all elements of harm 
        
        if label.root() == None:
            continue
        else:
            root = spellpc(label.root())
        
        chordtones = getchordtones(label)    
        
        notes = piece.flat.getElementsByClass(m21.note.Note)
        pitches = [spellpc(note.pitch) for note in notes
                   if getonset(note) < chord_offset and getoffset(note) > chord_onset] 
            
        notes = [(pitch-root, notetype(pitch, pitches, chordtones)) for pitch in pitches]
             
        harmonies.append((getchordtype(label), notes))
        
    return(harmonies)

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
    muspy.WikifoniaDataset(path.join("data", "wikifonia"))
    allchords = []

    print("extracting chords from pieces...")
    for file in tqdm.tqdm(glob.glob(path.join("data", "wikifonia", "Wikifonia", "*.mxl*"))):
        try:
            chords = readchords(file)
            allchords += chords
        except:
            print("Could not read chords from ", file)

    print("writing chords...")
    writechords(path.join("data", "wikifonia.tsv"), allchords)
    print("done.")
