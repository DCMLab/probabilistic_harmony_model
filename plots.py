# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: dcml-harmony-and-ornamentation
#     language: python
#     name: dcml-harmony-and-ornamentation
# ---

# %%
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({"text.usetex": True,
                     'text.latex.preamble' : r'\usepackage{amsmath,amssymb}',
                    })
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxPatch, BboxConnector, BboxConnectorPatch)
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import json
import utils
utils.set_fifth_range(14) # 2 diatonics above and below cente

from pathlib import Path


# %% [markdown]
# # Plots
#
# Plots of the model parameters for use in the paper.
#
# ## Loading Data

# %%
# load the data
def load_data(name):
    # chord types
    #df = utils.load_csv(path.join('data', name + '.tsv'))
    #sizes = df.groupby(['chordid', 'label']).size()
    #type_counts = sizes.groupby('label').size().sort_values(ascending=False)
    #chordtypes = type_counts.index.tolist()
    
    # parameters
    with open(Path('results', name+'_params.json'), 'r') as f:
        data = json.load(f)
    
    return {k:np.array(v) for k,v in data['params'].items()}, np.array(data['chordtypes'], dtype='object')


# %%
dcml_params, dcml_chordtypes = load_data('dcml')
# wiki_params, wiki_chordtypes = load_data('wikifonia')
ewld_params, ewld_chordtypes = load_data('ewld')
fifth_range = 14

# %%
ewld_chordtypes

# %%
dcml_map = {
    'M': 'major',
    'm': 'minor',
    'o': 'diminished',
    '+': 'augmented',
    'Mm7': 'dominant-7th',
    'mm7': 'minor-7th',
    'MM7': 'major-7th',
    'mM7': 'minor-major-7th',
    '%7': 'half-diminished',
    'o7': 'full-diminished',
    '+7': 'augmented-7th',
    'Ger': 'German-6th',
    'It': 'Italian-6th',
    'Fr': 'French-6th',
}
dcml_chordtypes_alt = np.vectorize(dcml_map.get)(dcml_chordtypes)
ewld_map = {
    'major': 'major',
    'dominant': 'dominant-7th',
    'minor': 'minor',
    'minor-seventh': 'minor-7th',
    'major-seventh': 'major-7th',
    'dominant-ninth': 'dominant-9th',
    'major-sixth': 'major-6th',
    'diminished': 'diminished',
    'minor-sixth': 'minor-6th',
    'half-diminished': 'half-diminished',
    'diminished-seventh': 'full-diminished',
    'augmented-seventh': 'augmented-7th',
    'augmented': 'augmented',
    'suspended-fourth': 'suspended-4th',
    'dominant-13th': 'dominant-13th',
    'minor-ninth': 'minor-9th',
    'major-ninth': 'major-9th',
    'dominant-11th': 'dominant-11th',
    'major-minor': 'minor-major-7th',
    'suspended-second': 'suspended-2nd',
    'minor-11th': 'minor-11th',
    'power': 'power',
    'major-13th': 'major-13th',
    'minor-13th': 'minor-13th'
}
ewld_chordtypes_alt = np.vectorize(ewld_map.get)(ewld_chordtypes)
chordtypes_common = np.array(
    ["major", "minor", "dominant-7th", "diminished",
     "full-diminished", "minor-7th", "half-diminished", "major-7th",
     "augmented", "minor-major-7th", "augmented-7th"],
    dtype="object")
chordtypes_all = np.array(['major', 'minor', 'dominant-7th', 'diminished', 'full-diminished',
       'minor-7th', 'half-diminished', 'major-7th', 'augmented',
       'German-6th', 'Italian-6th', 'French-6th', 'minor-major-7th',
       'augmented-7th', 'dominant-9th', 'major-6th', 'minor-6th',
       'suspended-4th', 'dominant-13th', 'minor-9th',
       'major-9th', 'dominant-11th', 'power',
       'suspended-2nd', 'minor-11th', 'major-13th', 'minor-13th'],
      dtype=object)


# %%
def saveplot(name, fig):
    fig.savefig(Path('plots', name+'.pdf'))


# %% [markdown]
# ## Posterior Plots

# %%
# posterior of 'rate_notes'
def plot_note_rate_post(ax, params, width=None):
    alpha = params['alpha_rate_notes']
    beta = params['beta_rate_notes']
    mean, var = stats.gamma.stats(alpha, scale=1/beta)
    print(f"{mean=}, {var=}")
    
    if width == None:
        limits = stats.gamma.interval(.999999, alpha, scale=1/beta)
    else:
        limits = (mean-width/2, mean+width/2)
    
    x = np.linspace(limits[0], limits[1], 400)
    y = stats.gamma.pdf(x, alpha, scale=1/beta)
    ax.plot(x, y)
    ax.set_xlim(limits)
    ax.set_xlabel('$\lambda$')

def plot_note_rates(params1, params2, width=None):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,3), sharey=True)
    plot_note_rate_post(ax1, params1, width=width)
    ax1.set_title("DCML")
    plot_note_rate_post(ax2, params2, width=width)
    ax2.set_title("EWLD")
    fig.tight_layout()
    return fig


# %%
note_rates = plot_note_rates(dcml_params, ewld_params, width=0.15)
saveplot('note_rates', note_rates)


# %%
# zoom effect
def connect_bbox(bbox1, bbox2, bbox3,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
        }

    c1 = BboxConnector(bbox1, bbox3, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox3, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)
    bbox_patch1.set_clip_on(False)
    bbox_patch2.set_clip_on(False)

    p = BboxConnectorPatch(bbox1, bbox3,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p
    
def zoom_effect(ax1, ax2, xmin, xmax, **kwargs):
    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)

    bbox = Bbox.from_extents(xmin, 0, xmax, -0.12)
    bbox2 = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, trans1)
    mybbox2 = TransformedBbox(bbox, trans2)
    mybbox3 = TransformedBbox(bbox2, trans2)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}#, "fc": "none"}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2, mybbox3,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

def zoom_effect2(ax1, ax2, xmin, xmax):
    
    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
    bbox = Bbox.from_extents(xmin, 0, xmax, -0.12)
    bbox2 = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, trans1)
    mybbox2 = TransformedBbox(bbox, trans2)
    mybbox3 = TransformedBbox(bbox2, trans2)

    prop_patches = {"ec": "none", "alpha": 0.2}
    prop_lines = {"ec": 'black', 'fc': 'none', 'alpha': 0.5}

    c1 = BboxConnector(mybbox1, mybbox3, loc1=3, loc2=2, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(mybbox1, mybbox3, loc1=4, loc2=1, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(mybbox1, **prop_patches)
    bbox_patch2 = BboxPatch(mybbox2, **prop_patches)
    bbox_patch3 = BboxPatch(mybbox3, **prop_lines)
    bbox_patch1.set_clip_on(False)
    bbox_patch2.set_clip_on(False)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(bbox_patch3)
    ax2.add_patch(c1)
    ax2.add_patch(c2)

    return c1, c2, bbox_patch1, bbox_patch2


# %%
# posterior of 'p_is_chordtone'
def plot_p_ict(ax, params, harmtypes, indices, lower=0, upper=1):
    alphas = np.array(params["alpha_p_ict"])[indices]
    betas  = np.array(params["beta_p_ict"])[indices]
    x = np.linspace(lower, upper, 400)
    y = np.array([stats.beta.pdf(x, a, b) for a, b in zip(alphas, betas)]).transpose()
    ax.plot(x,y)
    
    meansx = [stats.beta.mean(a,b) for a,b in zip(alphas,betas)]
    meansy = [stats.beta.pdf(x,a,b) for (x,a,b) in zip(meansx,alphas,betas)]
    for i,(xi,yi) in enumerate(zip(meansx,meansy)):
        ax.text(xi, yi+10, str(i+1), ha='center')
    
    ax.set_xlabel("$\\theta_h$")
    ax.set_xlim((lower, upper))
    ax.set_ylim((0,np.array(meansy).max()+50))

def plot_p_icts():
    selection_dcml = [np.where(dcml_chordtypes_alt == label)[0][0] for label in chordtypes_common[:-2]] # range(9)
    #selection_wiki = [0,2,1,7,11,3,9,4,13]
    selection_ewld = [np.where(ewld_chordtypes_alt == label)[0][0] for label in chordtypes_common[:-2]] # [0,2,1,7,10,3,9,4,12]
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(9,6))
    
    plot_p_ict(ax1, dcml_params, dcml_chordtypes_alt, selection_dcml,
               lower=0.74, upper=0.92)
    ax1.set_title('DCML', x=0.05, y=0.85)
        
    plot_p_ict(ax2, ewld_params, ewld_chordtypes_alt, selection_ewld,
               lower=0.62, upper=0.89)
    ax2.set_title('EWLD', x=0.05, y=0.85)
    
    ax1.legend([f"{i+1} - {dcml_chordtypes_alt[l]}" for (i,l) in enumerate(selection_dcml)],
               loc='upper right', ncol=2, framealpha=1)
    
    zoom_effect2(ax1,ax2,0.75,0.88)
    fig.tight_layout()
    return fig

icts = plot_p_icts()
saveplot('icts', icts)


# %%
# posterior of chord type probabilities
def plot_chord_type_dist(ax, params, labels):
    alphas = params['params_p_harmony']
    ax.barh(np.arange(len(alphas))[::-1], alphas, tick_label=labels)
    ax.set_xlabel("$\\alpha_h$")
    
def plot_chord_type_dists():
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,6))
    plot_chord_type_dist(ax1, dcml_params, dcml_chordtypes_alt)
    ax1.set_title("Chord Type Posterior (DCML)")
    plot_chord_type_dist(ax2, ewld_params, ewld_chordtypes_alt)
    ax2.set_title("Chord Type Posterior (EWLD)")
    fig.tight_layout()
    return fig
    
type_dist = plot_chord_type_dists()
saveplot('type_dist', type_dist)

# %%
interval_labels = ['dd1', 'dd5', 'd2', 'd6', 'd3', 'd7', 'd4',
                   'd1', 'd5', 'm2', 'm6', 'm3', 'm7', 'P4',
                   'P1', 'P5', 'M2', 'M6', 'M3', 'M7', 'a4',
                   'a1', 'a5', 'a2', 'a6', 'a3', 'a7', 'aa4', 'aa1']

def plot_profile(ax, params, i, name):
    chordtones = params['params_p_chordtones'][i]
    ornaments = params['params_p_ornaments'][i]
    #labels = [str(pitchtypes.SpelledIntervalClass(i)) for i in range(-fifth_range, fifth_range+1)]
    #labels = np.arange(-fifth_range, fifth_range+1)
    labels = interval_labels
    x = np.arange(2*fifth_range+1)
    width = 0.4
    ax.bar(x - width/2, chordtones, width, label='chord tones')
    ax.bar(x + width/2, ornaments, width, label='ornaments')
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical', va='top', family='monospace', usetex=False, size='small')
    ax.set_xlim((-1,2*fifth_range+1))
    #ax.legend()


# %%
def plot_example_profile():
    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    labels = interval_labels[7:22]
    chordtones = [1, 1, 1, 1, 1, 9, 1, 12, 11, 1, 1, 10, 1, 1, 1]
    ornaments = [1, 1, 1, 1, 1, 4, 6, 3, 1, 5, 3, 1, 1, 1, 1]
    x = np.arange(15)
    width=0.4
    ax.bar(x - width/2, chordtones, width, label='chord tones')
    ax.bar(x + width/2, ornaments, width, label='ornaments')
    ax.set_title("Example Profile")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical', va='top', family='monospace', usetex=False, size='small')
    ax.set_xlim((-1,15))
    ax.legend()
    return fig
    
example_profile = plot_example_profile()
saveplot('example_profile', example_profile)


# %%
# posteriors of note probabilities (for common chord types)
def plot_common_chords(indices):
    fig, axs = plt.subplots(len(indices), 2, figsize=(9,len(indices)*2))
    for i, ind in enumerate(indices):
        name = chordtypes_common[ind]
        plot_profile(axs[i,0], dcml_params, np.where(dcml_chordtypes_alt == name)[0][0], name+" (ABC+)")
        plot_profile(axs[i,1], ewld_params, np.where(ewld_chordtypes_alt == name)[0][0], name+" (EWLD)")
    axs[0,1].legend(labels=['chordtones', 'ornaments'], framealpha=1)
    fig.tight_layout()
    return fig
        
common1 = plot_common_chords([0,1,2,3,4,5])
saveplot('chordtypes_common1', common1)
common2 = plot_common_chords([6,7,8,9,10])
saveplot('chordtypes_common2', common2)


# %%
#plot remaining chords
def plot_rest_chords():
    rest_dcml = np.where(~(np.isin(dcml_chordtypes_alt, chordtypes_common)))[0]
    rest_wiki = np.where(~(np.isin(ewld_chordtypes_alt, chordtypes_common)))[0]
    
    #plot part 1
    fig1, axs1 = plt.subplots(3, 2, figsize=(9, 6))
    for i in range(3):
        ind = rest_dcml[i]
        plot_profile(axs1.flat[i], dcml_params, ind, dcml_chordtypes_alt[ind]+" (ABC+)")
    for i in range(3):
        ind = rest_wiki[i]
        plot_profile(axs1.flat[i+3], ewld_params, ind, ewld_chordtypes_alt[ind]+" (EWLD)")
    axs1[0,1].legend(labels=['chordtones', 'ornaments'], framealpha=1)
    fig1.tight_layout()
    saveplot('chordtypes_rest1', fig1)
    
    # plot part 2
    fig2, axs2 = plt.subplots(5, 2, figsize=(9, 10))
    for i in range(10):
        ind = rest_wiki[i+3]
        plot_profile(axs2.flat[i], ewld_params, ind, ewld_chordtypes_alt[ind]+" (EWLD)")
    axs2[0,1].legend(labels=['chordtones', 'ornaments'], framealpha=1)
    fig2.tight_layout()
    saveplot('chordtypes_rest2', fig2)

plot_rest_chords()


# %%
def plot_profile_small(ax, params, i, name):
    chordtones = params['params_p_chordtones'][i]
    ornaments = params['params_p_ornaments'][i]
    labels = np.arange(-fifth_range, fifth_range+1)
    x = np.arange(2*fifth_range+1)
    vals = np.stack([-chordtones, ornaments])
    extr = max(chordtones.max(), ornaments.max())
    ax.imshow(vals, cmap="bwr", vmin=-extr, vmax=extr)
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()


# %% [markdown]
# ## Clustering

# %%
data_path = Path('results', 'clustering')
device = 'cpu'

def mean_dist(X,Y):
    return abs(stats.beta.mean(X[0],X[1]) - stats.beta.mean(Y[0],Y[1]))

def replay_clustering(name, nharms, dist=mean_dist):
    # initial clustering
    cluster_assignment = dict((i,i) for i in range(nharms))

    # run iterative experiments
    outputs = []
    
    for it in range(nharms):
        nclusters = nharms - it
        print(f"iteration {it} ({nclusters} clusters).")
        print(cluster_assignment)
        
        # inference
        params_load = torch.load(data_path / f"{name}_params{nclusters}.pt")
        params = dict((n, torch.tensor(vals, device=device)) for n, vals in params_load.items())
        
        # record output
        outputs.append(dict({
            "params": params,
            "cluster_assignment": cluster_assignment,
        }))
        
        # compute next clustering / init
        if nclusters > 1:
            # find closest clusters
            alphas = params["alpha_p_ict"]
            betas = params["beta_p_ict"]
            dists = dict()
            for i in range(nclusters):
                for j in range(i+1, nclusters):
                    dists[(i,j)] = dist((alphas[i], betas[i]), (alphas[j], betas[j]))
            min1, min2 = min(dists, key=dists.get)

            # map clusters
            remaining = [i for i in range(nclusters) if i not in [min1,min2]]
            cluster_mapping = {**{min1: 0, min2: 0}, **dict((c,i+1) for i,c in enumerate(remaining))}
            
            # update assignment
            cluster_assignment = dict((h,cluster_mapping[c]) for h,c in cluster_assignment.items())

    return outputs

def load_cluster_params(name):
    with open(data_path / f"bf_{name}.json", 'r') as f:
        data = json.load(f)
    return data['chordtypes'], data['params']['params_model']


# %%
# the two datasets use a different order of chord types
dcml_chordtypes_common, dcml_cluster_probs = load_cluster_params('dcml')
dcml_outputs = replay_clustering("dcml", 9)
ewld_chordtypes_common, ewld_cluster_probs = load_cluster_params('ewld')
ewld_outputs = replay_clustering("ewld", 9)


# %%
def plot_cluster_probs(probs, name):
    n = len(probs)
    sns.barplot(x=[f"{n-i}" for i in range(n)], y=probs)
    plt.xlabel('number of clusters')
    plt.ylabel('model probability')
    plt.title(f"Cluster Model Probabilities ({name})")
    plt.show(block=False)

plot_cluster_probs(dcml_cluster_probs, "ABC+")
plot_cluster_probs(ewld_cluster_probs, "EWLD")

# %%
dcml_best_i, dcml_best_p = max(enumerate(dcml_cluster_probs), key=lambda x: x[1])
dcml_best_cluster = dcml_outputs[dcml_best_i]
ewld_best_i, ewld_best_p = max(enumerate(ewld_cluster_probs), key=lambda x: x[1])
ewld_best_cluster = ewld_outputs[ewld_best_i]
print(f"ABC+: {9-dcml_best_i} clusters, p={dcml_best_p}")
print(f"EWLD: {9-ewld_best_i} clusters, p={ewld_best_p}")


# %%
def plot_p_ict_cluster(ax, params, harmtypes, cluster_assignment, lower=0, upper=1):
    alphas = params["alpha_p_ict"]
    betas  = params["beta_p_ict"]
    #x = np.linspace(lower, upper, 600)
    #y = np.array([stats.beta.pdf(x, a, b) for a, b in zip(alphas, betas)]).transpose()
    means = alphas / (alphas + betas)
    
    # compute cluster names from assignment
    names = dict() # cluster index -> list of chord types
    for chord, cluster in cluster_assignment.items():
        name = harmtypes[chord]
        if(names.get(cluster) == None):
            names[cluster] = []
        names[cluster].append(name)
    # sort by cluster index
    its = sorted(names.items())
    name_labels = [str.join(", ",it[1]) for it in its]
    
    # combine all data and sort by cluster mean
    data = pd.DataFrame({"alpha": alphas, "beta": betas, "names": name_labels, "mean": means})
    data = data.sort_values("mean")
    x = np.linspace(lower, upper, 600)
    y = np.array([stats.beta.pdf(x, a, b) for a, b in zip(data.alpha, data.beta)]).transpose()
    
    # plot
    ax.plot(x,y)
    ax.set_xlabel(f"$\\theta$ ({len(set(cluster_assignment.values()))} clusters)")
    ax.legend(data.names, bbox_to_anchor=(1., 1),
              loc='upper left')

def plot_p_icts_clusters(dcml_cluster, ewld_cluster):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(9,6))
    
    plot_p_ict_cluster(ax1, dcml_cluster['params'], dcml_chordtypes_common, dcml_cluster['cluster_assignment'],
                       lower=0.735, upper=0.915)
    ax1.set_title('ABC+', x=0.05, y=0.85)
        
    plot_p_ict_cluster(ax2, ewld_cluster['params'], ewld_chordtypes_common, ewld_cluster['cluster_assignment'],
                       lower=0.62, upper=0.89)
    ax2.set_title('EWLD', x=0.05, y=0.85)
    
#     ax1.legend([f"{i+1} - {dcml_chordtypes_alt[i]}" for i in selection_dcml],
#                loc='upper right', ncol=2, framealpha=1)
    
    zoom_effect2(ax1,ax2,0.75,0.9)
    fig.tight_layout()
    return fig

clusters_plot = plot_p_icts_clusters(dcml_best_cluster, ewld_best_cluster)
saveplot('clusters', clusters_plot)
# %%
# for script mode
plt.show()


# %%
