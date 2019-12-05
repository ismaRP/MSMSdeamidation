from deamidation.MQreader import evidenceBatchReader
from deamidation.DSevidence import deamidationMatrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import deamidation.accFunctions as af
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

plt.style.use('seaborn-dark-palette')


main_path = '/home/ismael/palaeoproteomics/'
# Read radio carbon data
sampleInfo, header = af.readSampleInfo(main_path+'data/all_samples.tsv')
# Read proteins to filter
protsInfo = af.readProtList(main_path+'data/collagen.tsv')
# Read halftimes and properties
N_properties, Q_properties = af.readHalftimes(main_path+'data/N_properties.json',
                                              main_path+'data/Q_properties.json')
aa_properties = N_properties.copy()
aa_properties.update(Q_properties)

datapath = main_path+'datasets'
sf_exp = {'pompeii_cph', 'pompeii2_cph'}
out_dir = main_path+'/out/'


key = 'Substrate'
type = 'cat'


reader = evidenceBatchReader(datapath, byPos=True, qt=False, sf_exp=sf_exp)

data = reader.readBatch()

sampleTripeps = data.get_sampleTripeps(sampleInfo, protsInfo, norm_type='simple',
                                       filter=None)

deamid_mat = deamidationMatrix(sampleTripeps, sampleInfo, header)


sums = np.sum(deamid_mat.counts, axis=0)
maxs = np.max(deamid_mat.counts, axis=0)
max_cond = maxs > 8
upper_sum = sums > 20 # Keep
mid_sum = np.logical_and(20 >= sums, sums > 10) # Keep if max is high enough
mid_sum = np.logical_and(mid_sum, max_cond)
tripep_mask = np.logical_or(upper_sum, mid_sum)
deamid_mat = deamid_mat.filter_tripeps(tripep_mask)

merged_deamid_mat = deamid_mat.merge_by_tripep()
# merged_deamid_mat = merged_deamid_mat.filter_tripeps(merged_deamid_mat.filter_by_pwcounts())
# man_rm = ['MQG', 'HQG', 'PQL', 'DNG', 'GQH', 'GNN', 'NNG']

# Impute data on merged_deamid_mat
# Imputing values

influx, prop_missing = merged_deamid_mat.influx()
influx_filter = influx > 0.5
prop_missing_filter = prop_missing < 0.6
merged_deamid_mat = merged_deamid_mat.filter_tripeps(prop_missing_filter)

imp = IterativeImputer(max_iter=500, random_state=0)
D_imp = imp.fit_transform(merged_deamid_mat.D)
print('IterativeImputer: number of iterations: {}'.format(imp.n_iter_))

# PCA
D_imp_pca, _, sorted_evals = af.pca(D_imp, 3)


pc_var = 100*np.cumsum(sorted_evals/np.sum(sorted_evals))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,len(sorted_evals)+1), pc_var, 'o-')
plt.xlabel('PC')
plt.ylabel('% Cumulative normalized variance')
plt.title('Cumulative normalized variance across PCs')
plt.savefig(out_dir + 'pca_var.png')
plt.close()


# -------------------------------------------------------------------
Yvect = merged_deamid_mat.Ydata['Substrate']
Yset = set(Yvect)
Yset.remove('Gelatine')
Yset.remove('Glue')
Yset = np.sort(list(Yset))

# Map to color
keyCm = plt.get_cmap('tab20')
keyNorm  = plt.Normalize(vmin=0, vmax=2*len(Yset))
keySm = ScalarMappable(norm=keyNorm, cmap=keyCm)
keySm.set_array([])
map_color = {}
i = 1
for Yval in Yset:
    map_color[Yval] = keySm.to_rgba(i)
    i += 1

# Add gelatine and glue manually
map_color['Gelatine'] = 'blue'
map_color['Glue'] = 'pink'
Yset = np.append(Yset, 'Gelatine')
Yset = np.append(Yset, 'Glue')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,17), dpi=300)
# axes[0,0] PC1 vs PC2 by substrate
# axes[0,1] PC1 vs PC3 by substrate
# axes[1,0] PC1 vs PC2 by age
# axes[1,1] PC1 vs PC3 by age

## PLOT BY SUBSTRATE
for sub in Yset:
    mask = Yvect == sub
    axes[0,0].scatter(D_imp_pca[mask,0], D_imp_pca[mask,1], c=map_color[sub],
                 alpha=0.7, label=sub, s=60)
    axes[0,1].scatter(D_imp_pca[mask,0], D_imp_pca[mask,2], c=map_color[sub],
                 alpha=0.7, label=sub, s=60)

axes[0,0].legend(loc=9, bbox_to_anchor=(1.13, 1.17), ncol=4, fontsize = 'xx-large')
# fig.suptitle('PCA plot')
# ax1.set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)))
axes[0,0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')
# ax2.set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)))
axes[0,1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')

## PLOT BY AGE
ages = merged_deamid_mat.Ydata['Age'].astype('float')
known = ages != -1
# Logscale ages
ages[known] = np.log(ages[known]+1)

htcm = plt.get_cmap("cool")
htnorm = plt.Normalize(np.min(ages[known]), np.max(ages[known]))
htsm = ScalarMappable(norm=htnorm, cmap=htcm)
htsm.set_array([])
# plot unknown age
axes[1,0].scatter(D_imp_pca[~known,0], D_imp_pca[~known,1], c='lightgrey',
             alpha=0.7)
axes[1,1].scatter(D_imp_pca[~known,0], D_imp_pca[~known,2], c='lightgrey',
             alpha=0.7)
# plot known age
axes[1,0].scatter(D_imp_pca[known,0], D_imp_pca[known,1], c=htsm.to_rgba(ages[known]),
             alpha=0.9, s=60)
axes[1,1].scatter(D_imp_pca[known,0], D_imp_pca[known,2], c=htsm.to_rgba(ages[known]),
             alpha=0.9, s=60)
htcbar_ax = fig.add_axes([0.3, 0.47, 0.4, 0.015])
htcbar = plt.colorbar(htsm, cax=htcbar_ax, orientation="horizontal")
htcbar.ax.set_title(r"$\log(YBP+1)$", size='x-large')
htcbar.ax.tick_params(labelsize='x-large', length=4)
axes[1,0].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1,0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1,1].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1,1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')
plt.subplots_adjust(
    left = 0.1,  # the left side of the subplots of the figure
    right = 0.93 ,  # the right side of the subplots of the figure
    bottom = 0.08 , # the bottom of the subplots of the figure
    top = 0.9  ,   # the top of the subplots of the figure
    wspace = 0.2 , # the amount of width reserved for space between subplots,
                   # expressed as a fraction of the average axis width
    hspace = 0.28,  # the amount of height reserved for space between subplots,
                   # expressed as a fraction of the average axis height
)

for r in axes:
    for ax in r:
        ax.tick_params(labelsize='x-large', length=4)

plt.savefig(out_dir + 'PCA_comb.png')
plt.close()


## PLOT BY AGE ONLY BONE
print('Plotting bone samples')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5), dpi=300)
bone_samples = ['Bone', 'Tar seep bone']
markers =  ['.', '^']
ages = merged_deamid_mat.Ydata['Age'].astype('float')
ages_b = ages[np.logical_or(Yvect=='Bone', Yvect=='Tar seep bone')]
known = ages_b != -1
# Logscale ages
ages_b[known] = np.log(ages_b[known]+1)
# Remap color
htcm = plt.get_cmap("cool")
htnorm = plt.Normalize(np.min(ages_b[known]), np.max(ages_b[known]))
htsm = ScalarMappable(norm=htnorm, cmap=htcm)
htsm.set_array([])
for s, m in zip(bone_samples, markers):
    mask = Yvect == s
    D_s = D_imp_pca[mask,:]
    ages_s = ages[Yvect==s]
    known_s = ages_s != -1
    ages_s[known_s] = np.log(ages_s[known_s]+1)
    # Plot unknown age
    axes[0].scatter(D_s[~known_s,0], D_s[~known_s,1], c='lightgrey',
                 alpha=0.7, marker=m)
    axes[1].scatter(D_s[~known_s,0], D_s[~known_s,2], c='lightgrey',
                 alpha=0.7, marker=m)
    # Plot known age
    axes[0].scatter(D_s[known_s,0], D_s[known_s,1], c=htsm.to_rgba(ages_s[known_s]),
                 alpha=0.9, s=80, marker=m)
    axes[1].scatter(D_s[known_s,0], D_s[known_s,2], c=htsm.to_rgba(ages_s[known_s]),
                 alpha=0.9, s=80, marker=m, label=s)


axes[1].legend(fontsize = 'x-large')
# bbox_to_anchor=(1.13, 1.17)
htcbar_ax = fig.add_axes([0.3, 0.93, 0.4, 0.015])
htcbar = plt.colorbar(htsm, cax=htcbar_ax, orientation="horizontal")
htcbar.ax.set_title(r"$\log(YBP+1)$", size='x-large')
htcbar.ax.tick_params(labelsize='x-large', length=4)

axes[0].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')

for ax in axes:
    ax.tick_params(labelsize='x-large', length=4)

plt.savefig(out_dir + 'PCA_bone_tarseep.pdf', format='pdf')
plt.close()


## PLOT BY THERMAL AGE ONLY BONE
print('Plotting bone samples')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5), dpi=300)
bone_samples = ['Bone', 'Tar seep bone']
markers =  ['.', '^']
ages = merged_deamid_mat.Ydata['10C Thermal age'].astype('float')
ages_b = ages[np.logical_or(Yvect=='Bone', Yvect=='Tar seep bone')]
known = ages_b != -1
# Logscale ages
ages_b[known] = np.log(ages_b[known]+1)
# Remap color
htcm = plt.get_cmap("cool")
htnorm = plt.Normalize(np.min(ages_b[known]), np.max(ages_b[known]))
htsm = ScalarMappable(norm=htnorm, cmap=htcm)
htsm.set_array([])
for s, m in zip(bone_samples, markers):
    mask = Yvect == s
    D_s = D_imp_pca[mask,:]
    ages_s = ages[Yvect==s]
    known_s = ages_s != -1
    ages_s[known_s] = np.log(ages_s[known_s]+1)
    # Plot unknown age
    axes[0].scatter(D_s[~known_s,0], D_s[~known_s,1], c='lightgrey',
                 alpha=0.7, marker=m)
    axes[1].scatter(D_s[~known_s,0], D_s[~known_s,2], c='lightgrey',
                 alpha=0.7, marker=m)
    # Plot known age
    axes[0].scatter(D_s[known_s,0], D_s[known_s,1], c=htsm.to_rgba(ages_s[known_s]),
                 alpha=0.9, s=80, marker=m)
    axes[1].scatter(D_s[known_s,0], D_s[known_s,2], c=htsm.to_rgba(ages_s[known_s]),
                 alpha=0.9, s=80, marker=m, label=s)


axes[1].legend(fontsize = 'x-large')
# bbox_to_anchor=(1.13, 1.17)
htcbar_ax = fig.add_axes([0.3, 0.93, 0.4, 0.015])
htcbar = plt.colorbar(htsm, cax=htcbar_ax, orientation="horizontal")
htcbar.ax.set_title(r"$\log(YBP+1)$", size='x-large')
htcbar.ax.tick_params(labelsize='x-large', length=4)

axes[0].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')

for ax in axes:
    ax.tick_params(labelsize='x-large', length=4)

plt.savefig(out_dir + 'PCA_ThermalAge_bone_tarseep.pdf', format='pdf')
plt.close()
