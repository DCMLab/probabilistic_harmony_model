from IPython.display import Audio, display
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# data preprocessing
# ------------------

def isneighbour(p1,p2):
    return abs(p1-p2)%7 in [2,5]

def notetype(pitch, pitches, chordtones):
    hasneighbour = any(isneighbour(p, pitch) for p in pitches) # any uses iterator here
    hasctneighbour = any(isneighbour(p, pitch) and (p in chordtones) for p in pitches) 
    
    if not hasneighbour: # if there is no identifiable neighbour, tone has to be chord tone
        return 'chordtone'
    elif pitch in chordtones and not hasctneighbour:
        return 'chordtone'
    elif not pitch in chordtones and hasctneighbour:
        return 'ornament'
    else:
        return 'unknown'

# pitch class range
# -----------------

fifth_range = 2*7

def set_fifth_range(f):
    global fifth_range
    fifth_range = f

def get_fifth_range():
    return fifth_range

def get_npcs():
    return 2*fifth_range + 1

def fifth_to_index(fifth):
    """Turns a LoF pitch class into an index."""
    return fifth + fifth_range

def index_to_fifth(index):
    """Turns an index into a LoF pitch class"""
    return index - fifth_range

def chord_tensor(fifths, types):
    """Takes a list of notes as fifths and a list of corresponding note types."""
    notetype = {'chordtone': 0, 'ornament': 1, 'unknown': 2}
    chord = torch.zeros((3, get_npcs()))
    for (fifth, t) in zip(fifths, types):
        chord[notetype[t], fifth_to_index(fifth)] += 1
    return chord.reshape((1,-1))

# loading data
# ------------

def load_csv(fn, sep='\t'):
    df = pd.read_csv(fn, sep=sep).dropna()
    # some notes are too far away from the root for our model, so we drop them
    in_range = (df['fifth'] >= -fifth_range) & (df['fifth'] <= fifth_range)
    return df[in_range]

# plotting etc.
# -------------

def play_chord(amplitudes, T=2, sr=22050):
    pcs = np.arange(-fifth_range, fifth_range+1)
    tones = 262. * np.exp(pcs * np.log(1.5) % np.log(2))
    tones = np.concatenate((0.5*tones, tones, 2*tones))
    n = int(T*sr)
    t = np.linspace(0, T, n)
    sines = np.sin(2 * np.pi * tones[:, np.newaxis] * t[np.newaxis, :])
    amps = amplitudes / amplitudes.sum()
    mix = 0.1 * np.dot(amps.repeat(3), sines) # ear safety first
#     plt.plot(mix[0:1000])
#     plt.show()
    display(Audio(mix, rate=sr, normalize=False))


def plot_profile(chordtones, ornaments, name):
    labels = np.arange(-fifth_range, fifth_range+1)
    x = np.arange(get_npcs())
    width = 0.4
    fig, ax = plt.subplots(figsize=(15,5))
    plt.bar(x - width/2, chordtones, width, label='chord tones')
    plt.bar(x + width/2, ornaments, width, label='ornaments')
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()
