from deamidation.MQreader import evidenceBatchReader
from deamidation.DSevidence import deamidationMatrix
import deamidation.deamidCorr as dc
import deamidation.accFunctions as af
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
import os
import warnings
exit()
plt.style.use('seaborn-dark-palette')
warnings.simplefilter("ignore")
def sort_by_hts(halftimes, deamid_mat):
    hts = np.array([halftimes[trp]['halftime'] if trp in halftimes else -1
                    for trp in deamid_mat.trps_data['tripep']])
    idx = np.argsort(hts)
    hts = hts[idx]
    deamid_mat = deamid_mat.sort_tripeps(idx)
    return deamid_mat, hts

# Read radio carbon data
sampleInfo, header = af.readSampleInfo('./data/all_samples.tsv')
# Read proteins to filter
protsInfo = af.readProtList('./data/collagen.tsv')
# Read halftimes and properties
N_halftimes, Q_halftimes = af.readHalftimes('./data/N_properties.json',
                                   './data/Q_properties.json')
halftimes = N_halftimes.copy()
halftimes.update(Q_halftimes)

datapath = '/home/ismael/palaeoproteomics/datasets'
sf_exp = {'pompeii_cph', 'pompeii_cph_semitryptic', 'pompeii2_cph'}
exclude_data = {'gelatin_fish'}

out_dir = '/home/ismael/palaeoproteomics/out/'


key = 'Substrate'
type = 'cat'


reader = evidenceBatchReader(datapath, byPos=True, qt=False, sf_exp=sf_exp,
                             exclude=exclude_data)

data = reader.readBatch()

sampleTripeps = data.get_sampleTripeps(sampleInfo, protsInfo, norm_type='simple',
                                       filter=None)

deamid_mat = deamidationMatrix(sampleTripeps, sampleInfo, header)

# Filter out non-helical positions
deamid_mat = deamid_mat.filter_tripeps(deamid_mat.trps_data['position'] < 1193)

# Remove glue samples
samples_filter = deamid_mat.Ydata[key] != 'Glue'
deamid_mat = deamid_mat.filter_samples(samples_filter)


# Correct positions
deamid_mat.correct_pos(protsInfo)
# Merge by position
deamid_mat = deamid_mat.merge_by_pos(corr=True)
# Apply period
n_period = deamid_mat.stagger_pos(l_period=234)
# Sort by corr_pos
idx = np.argsort(deamid_mat.trps_data, order=['corr_pos'])
deamid_mat = deamid_mat.sort_tripeps(idx)
n_period = n_period[idx]

# COL1A1 filter
col1a1_filter = deamid_mat.trps_data['prot_name'] == 'COL1A1'
deamid_mat = deamid_mat.filter_tripeps(col1a1_filter)
n_period = n_period[col1a1_filter]

# Keep only non-zero corrected position
nonz_filter = deamid_mat.trps_data['corr_pos'] > 0
deamid_mat = deamid_mat.filter_tripeps(nonz_filter)
n_period = n_period[nonz_filter]

# Split XNY and XQY tripeptides
Qfilter = [True if trp[1]=='Q' else False
            for trp in deamid_mat.trps_data['tripep']]
deamid_matQ = deamid_mat.filter_tripeps(Qfilter)
n_periodQ = n_period[Qfilter]

Nfilter = [True if trp[1]=='N' else False
            for trp in deamid_mat.trps_data['tripep']]
deamid_matN = deamid_mat.filter_tripeps(Nfilter)
n_periodN = n_period[Nfilter]

###### Plot functions, attributes and variables
# Window
def window(x, y, size):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    x = list(x)
    y = list(y)
    assert len(x) == len(y)
    x_result = x[0:size]
    y_result = y[0:size]
    if len(x_result) == size:
        yield np.nanmedian(x_result), np.nanmean(y_result)
    for i in range(size, len(x)):
        x_result = x_result[1:] + [x[i]]
        y_result = y_result[1:] + [y[i]]
        yield np.nanmedian(x_result), np.nanmean(y_result)
bbta = (0,1.12) # bbox_to_anchor

fontP = FontProperties()
fontP.set_size('x-large')



####### Plot by substrate

Yvect = deamid_mat.Ydata[key]
Yset = np.sort(list(set(Yvect)))
keyCm = plt.get_cmap('tab20')
keyNorm  = plt.Normalize(vmin=0, vmax=2*len(Yset))
keySm = ScalarMappable(norm=keyNorm, cmap=keyCm)
keySm.set_array([])
mean_color = {}
i = 1
for Yval in Yset:
    mean_color[Yval] = keySm.to_rgba(i)
    i += 1

handles = []
for Yval in Yset:
    # Handles for N
    handles.append(plt.plot([], [], markersize=10, marker='.',
                       ls='', c = mean_color[Yval])[0])
    # Handles for Q
    handles.append(plt.plot([], [], markersize=10, marker='^',
                       ls='', c = mean_color[Yval])[0])
plt.close()

labels = []
fig = plt.figure(figsize=(18, 9), dpi=300)
ax = fig.add_subplot(111)
for Yval in Yset:
    sele = Yvect == Yval

    avgN = np.nanmean(deamid_matN.D[sele,:], axis=0)
    posN = deamid_matN.trps_data['corr_pos']
    ax.plot(posN, avgN, c=mean_color[Yval],
             marker='.', ls='', ms=12, alpha=0.8)
    labels.append(Yval + ' | XNZ')

    avgQ = np.nanmean(deamid_matQ.D[sele,:], axis=0)
    posQ = deamid_matQ.trps_data['corr_pos']
    ax.plot(posQ, avgQ, c=mean_color[Yval],
             marker='^', ls='', ms=12, alpha=0.8)
    labels.append(Yval + ' | XQZ')

ax.legend(handles, labels, ncol=5, loc='upper left', prop=fontP,
          bbox_to_anchor=bbta, fancybox=True)
ax.set_xlabel('Periodic position', size='x-large')
ax.set_ylabel('Deamidation', size='x-large')
ax.tick_params(labelsize='x-large', length=4)
fig.tight_layout()
fig.savefig(out_dir + 'stagger_pos.png')
plt.close()

handles = []
w_size = 20
w_step = 1
for Yval in Yset:
    # Handles for N
    handles.append(plt.plot([], [], markersize=10, marker='',
                            ls='--', c = mean_color[Yval])[0])
    # Handles for Q
    handles.append(plt.plot([], [], markersize=10, marker='',
                            ls='-', c = mean_color[Yval])[0])
plt.close()

labels = []
fig = plt.figure(figsize=(18, 9), dpi=300)
ax = fig.add_subplot(111)
for Yval in Yset:
    sele = Yvect == Yval
    # For N
    posN = deamid_matN.trps_data['corr_pos']
    avgN = np.nanmean(deamid_matN.D[sele,:], axis=0)
    a = window(posN, avgN, size=w_size)
    x_w = []
    y_w = []
    for k in a:
        x_w.append(k[0]); y_w.append(k[1])
    ax.plot(x_w, y_w, c=mean_color[Yval],
             marker='', ls='--', lw=2, alpha=0.8)
    labels.append(Yval + ' | XNZ')
    # For Q
    posQ = deamid_matQ.trps_data['corr_pos']
    avgQ = np.nanmean(deamid_matQ.D[sele,:], axis=0)
    a = window(posQ, avgQ, size=w_size)
    x_w = []
    y_w = []
    for k in a:
        x_w.append(k[0]); y_w.append(k[1])
    ax.plot(x_w, y_w, c=mean_color[Yval],
             marker='', ls='-', lw=2, alpha=0.8)
    labels.append(Yval + ' | XQZ')


ax.legend(handles, labels, ncol=5, loc='upper left', prop=fontP,
          bbox_to_anchor=bbta, fancybox=True)
ax.set_xlabel('Periodic position', size='x-large')
ax.set_ylabel('Deamidation', size='x-large')
ax.tick_params(labelsize='x-large', length=4)
fig.tight_layout()
fig.savefig(out_dir + 'window_stagger_pos.png')
plt.close()


##### Plot period by period

keyCm = plt.get_cmap('tab10')
keyNorm  = plt.Normalize(vmin=0, vmax=2*np.max(n_period))
keySm = ScalarMappable(norm=keyNorm, cmap=keyCm)
keySm.set_array([])


handles = []
for p in range(1, np.max(n_period)+1):
    # Handles for N
    handles.append(plt.plot([], [], markersize=10, marker='.',
                            ls='', c = keySm.to_rgba(p))[0])
    # Handles for Q
    handles.append(plt.plot([], [], markersize=10, marker='^',
                            ls='', c = keySm.to_rgba(p))[0])
plt.close()
w_size = 10
labels = []
fig = plt.figure(figsize=(18, 9), dpi=300)
ax = fig.add_subplot(111)
for p in range(1, np.max(n_period)+1):
    for Yval in Yset:
        s_sele = Yvect == Yval
        p_seleN = n_periodN == p
        posN = deamid_matN.trps_data['corr_pos'][p_seleN]
        matN = deamid_matN.D[:,p_seleN]
        matN = matN[s_sele,:]
        avgN = np.nanmean(matN, axis=0)
        ax.plot(posN, avgN, c=keySm.to_rgba(p),
                 marker='.', ls='', ms=12, alpha=0.8)

        p_seleQ = n_periodQ == p
        posQ = deamid_matQ.trps_data['corr_pos'][p_seleQ]
        matQ = deamid_matQ.D[:,p_seleQ]
        matQ = matQ[s_sele,:]
        avgQ = np.nanmean(matQ, axis=0)
        ax.plot(posQ, avgQ, c=keySm.to_rgba(p),
                 marker='^', ls='', ms=12, alpha=0.8)
    labels.append('Period {} | XNZ'.format(p))
    labels.append('Period {} | XQZ'.format(p))

ax.legend(handles, labels, ncol=6, loc='upper left', prop=fontP,
          bbox_to_anchor=bbta, fancybox=True)
ax.set_xlabel('Periodic position', size='x-large')
ax.set_ylabel('Deamidation', size='x-large')
ax.tick_params(labelsize='x-large', length=4)
fig.tight_layout()
fig.savefig(out_dir + 'stagger_period.png')
plt.close()


# Windows
w_size = 10
labels = []
fig = plt.figure(figsize=(18,9), dpi=300)
ax = fig.add_subplot(111)
for p in range(1, np.max(n_period)+1):
    avg_pN = []
    pos_pN = []
    avg_pQ = []
    pos_pQ = []
    for Yval in Yset:
        s_sele = Yvect == Yval
        p_seleN = n_periodN == p
        posN = deamid_matN.trps_data['corr_pos'][p_seleN]
        pos_pN += list(posN)
        matN = deamid_matN.D[:,p_seleN]
        matN = matN[s_sele,:]
        avgN = np.nanmean(matN, axis=0)
        avg_pN += list(avgN)

        p_seleQ = n_periodQ == p
        posQ = deamid_matQ.trps_data['corr_pos'][p_seleQ]
        pos_pQ += list(posQ)
        matQ = deamid_matQ.D[:,p_seleQ]
        matQ = matQ[s_sele,:]
        avgQ = np.nanmean(matQ, axis=0)
        avg_pQ += list(avgQ)
    idxN = np.argsort(pos_pN)
    pos_pN = np.array(pos_pN)[idxN]
    avg_pN = np.array(avg_pN)[idxN]
    idxQ = np.argsort(pos_pQ)
    pos_pQ = np.array(pos_pQ)[idxQ]
    avg_pQ = np.array(avg_pQ)[idxQ]
    a = window(pos_pN, avg_pN, size=w_size)
    x_w = []
    y_w = []
    for k in a:
        x_w.append(k[0]); y_w.append(k[1])
    ax.plot(x_w, y_w, c=keySm.to_rgba(p),
             marker='', ls='--', lw=2, alpha=0.8)
    labels.append('Period {} | XQZ'.format(p))
    a = window(pos_pQ, avg_pQ, size=w_size)
    x_w = []
    y_w = []
    for k in a:
        x_w.append(k[0]); y_w.append(k[1])
    ax.plot(x_w, y_w, c=keySm.to_rgba(p),
             marker='', ls='-', lw=2, alpha=0.8)
    labels.append('Period {} | XQZ'.format(p))
ax.legend(handles, labels, ncol=6, loc='upper left', prop=fontP,
          bbox_to_anchor=bbta, fancybox=True)
ax.set_xlabel('Periodic position', size='x-large')
ax.set_ylabel('Deamidation', size='x-large')
ax.tick_params(labelsize='x-large', length=4)
fig.tight_layout()
fig.savefig(out_dir + 'window_stagger_period.png')
plt.close()
