from deamidation.MQreader import evidenceBatchReader
from deamidation.DSevidence import deamidationMatrix
import deamidation.reactionRates as rr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import deamidation.accFunctions as af
from scipy.stats import spearmanr
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

plt.style.use('seaborn-deep')

def sort_by_lambda(properties, deamid_mat):
    lambdas = np.array([properties[trp]['lambda'] if trp in properties else -1
                        for trp in deamid_mat.trps_data['tripep']])
    idx = np.argsort(lambdas)
    lambdas = lambdas[idx]
    deamid_mat = deamid_mat.sort_tripeps(idx)
    return deamid_mat, lambdas


main_path = '/home/ismael/palaeoproteomics/MSMSdatasets/'
# Read radio carbon data
# sampleInfo, header = af.readSampleInfo(main_path+'data/all_samples.tsv')
sampleInfo, header = af.readSampleInfo(main_path+'tarseep_bone_dentalcalc_samples.tsv')

# Read proteins to filter
protsInfo = af.readProtList(main_path+'collagen.tsv')
# Read halftimes and properties
N_properties, Q_properties = af.readHalftimes(main_path+'N_properties.json',
                                              main_path+'Q_properties.json')
aa_properties = N_properties.copy()
aa_properties.update(Q_properties)

datapath = main_path + 'mq'
sf_exp = {'pompeii_cph', 'pompeii2_cph'}
out_dir = main_path + '/out/'
base_name = 'logtr_bone'

key = 'Substrate'
type = 'cat'


reader = evidenceBatchReader(datapath, byPos=True, tr='log', sf_exp=sf_exp)

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

# Filter out pompeii samples
pompeii_samples = {'pompeii cph', 'pompeii2 cph', 'big pompeii'}
# pompeii_samples = {}
filter = [False if d in pompeii_samples else True
          for d in deamid_mat.Ydata['Dataset']]

deamid_mat = deamid_mat.filter_samples(filter)

# man_rm = ['MQG', 'HQG', 'PQL', 'DNG', 'GQH', 'GNN', 'NNG']
# man_filter = [True if trp not in man_rm else False
#               for trp in merged_deamid_mat.trps_data['tripep']]
# merged_deamid_mat = merged_deamid_mat.filter_tripeps(man_filter)
influx, prop_missing = merged_deamid_mat.influx()
influx_filter = influx > 0.5
prop_missing_filter = prop_missing < 0.6
merged_deamid_mat = merged_deamid_mat.filter_tripeps(prop_missing_filter)


# Split
Qfilter = [True if trp[1]=='Q' else False
            for trp in merged_deamid_mat.trps_data['tripep']]
deamid_matQ = merged_deamid_mat.filter_tripeps(Qfilter)

Nfilter = [True if trp[1]=='N' else False
            for trp in merged_deamid_mat.trps_data['tripep']]
deamid_matN = merged_deamid_mat.filter_tripeps(Nfilter)

deamid_matQ, lambdasQ = sort_by_lambda(aa_properties, deamid_matQ)
deamid_matN, lambdasN = sort_by_lambda(aa_properties, deamid_matN)


dataR_N = rr.calc_Ri(
    deamid_matN.D,
    deamid_matN.counts,
    low_counts=4,
    log_tr=True,
    num_tol=0
)

dataR_Q = rr.calc_Ri(
    deamid_matQ.D,
    deamid_matQ.counts,
    low_counts=4,
    log_tr=True,
    num_tol=0
)

argsortN = np.argsort([np.median(v) for v in dataR_N])
argsortQ = np.argsort([np.median(v) for v in dataR_Q])
corrN = spearmanr(argsortN, np.array(range(len(argsortN)))).correlation
corrQ = spearmanr(argsortQ, np.array(range(len(argsortQ)))).correlation
print('All samples, XNZ, rho={}'.format(corrN))
print('All samples, XQZ, rho={}'.format(corrQ))

R_Nfig = rr.Rlambda_distr(dataR_N, deamid_matN.trps_data,
                          sort_by=None, log=True,
                          return_pl=True)

R_Qfig = rr.Rlambda_distr(dataR_Q, deamid_matQ.trps_data,
                          sort_by=None, log=True,
                          return_pl=True)


# Impute data on merged_deamid_mat
# Imputing values

imp = IterativeImputer(max_iter=500, random_state=0)
D_imp = imp.fit_transform(merged_deamid_mat.D)
print('IterativeImputer: number of iterations: {}'.format(imp.n_iter_))

# PCA

D_imp_pca, sorted_evecs, sorted_evals = af.pca(D_imp, 5, inv=True)
sorted_evecs = sorted_evecs[:,0:5]

# Plot PC variance
pc_var = 100*np.cumsum(sorted_evals/np.sum(sorted_evals))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,len(sorted_evals)+1), pc_var, 'o-')
plt.xlabel('PC')
plt.ylabel('% Cumulative normalized variance')
plt.title('Cumulative normalized variance across PCs')
plt.savefig(out_dir + base_name + '_pca_var.png')
plt.close()

# Plot loadings contributions
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5), dpi=300)
for i,trp in enumerate(merged_deamid_mat.trps_data['tripep']):
    axes[0].arrow(0, 0,
                  sorted_evecs[i,0]*np.sqrt(sorted_evals[0]),
                  sorted_evecs[i,1]*np.sqrt(sorted_evals[1]),
                  color = 'g', alpha=0.7)
    axes[0].text(sorted_evecs[i,0] * 1.1 * np.sqrt(sorted_evals[0]),
                 sorted_evecs[i,1] * 1.1 * np.sqrt(sorted_evals[1]),
                 trp,  color = 'black', ha = 'center', va = 'center',
                 fontsize=8)
tmp_sorted_evecs = np.vstack((sorted_evecs, np.zeros((1,5)))) # Ensure 0,0 appears in the plot
axes[0].set_xlim(np.min(tmp_sorted_evecs[:,0]*np.sqrt(sorted_evals[0]))-0.1,
                 np.max(tmp_sorted_evecs[:,0]*np.sqrt(sorted_evals[0]))+0.1)
axes[0].set_ylim(np.min(tmp_sorted_evecs[:,1]*np.sqrt(sorted_evals[1]))-0.1,
                 np.max(tmp_sorted_evecs[:,1]*np.sqrt(sorted_evals[1]))+0.1)
axes[0].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')

for i,trp in enumerate(merged_deamid_mat.trps_data['tripep']):
    axes[1].arrow(0, 0,
                  sorted_evecs[i,0]*np.sqrt(sorted_evals[0]),
                  sorted_evecs[i,2]*np.sqrt(sorted_evals[2]),
                  color = 'g', alpha=0.7)
    axes[1].text(sorted_evecs[i,0] * 1.1 * np.sqrt(sorted_evals[0]),
                 sorted_evecs[i,2] * 1.1 * np.sqrt(sorted_evals[2]),
                 trp, color = 'black', ha = 'center', va = 'center',
                 fontsize=8)
axes[1].set_xlim(np.min(tmp_sorted_evecs[:,0]*np.sqrt(sorted_evals[0]))-0.1,
                 np.max(tmp_sorted_evecs[:,0]*np.sqrt(sorted_evals[0]))+0.1)
axes[1].set_ylim(np.min(tmp_sorted_evecs[:,2]*np.sqrt(sorted_evals[2]))-0.1,
                 np.max(tmp_sorted_evecs[:,2]*np.sqrt(sorted_evals[2]))+0.1)
axes[1].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')
plt.savefig(out_dir + base_name + '_eigenvectors.png', format='png')
plt.close()
# Extract loadings for the first 5 PCs and
# for N and Q tripeps in the right order
Npos = np.array([
    np.where(merged_deamid_mat.trps_data['tripep'] == trpN)[0]
    for trpN in deamid_matN.trps_data['tripep']
])
Qpos = np.array([
    np.where(merged_deamid_mat.trps_data['tripep'] == trpQ)[0]
    for trpQ in deamid_matQ.trps_data['tripep']
])

Nloadings = sorted_evecs[Npos, np.array([0,1,2,3,4])] * np.sqrt(sorted_evals[0:5])
Qloadings = sorted_evecs[Qpos, np.array([0,1,2,3,4])] * np.sqrt(sorted_evals[0:5])

R_Nax2 = R_Nfig.axes[0].twinx()
R_Qax2 = R_Qfig.axes[0].twinx()

Nlines = R_Nax2.plot(np.arange(1, Nloadings.shape[0]+1), Nloadings,
                     lw=4, alpha=0.8, zorder=1)
Qlines = R_Qax2.plot(np.arange(1, Qloadings.shape[0]+1), Qloadings,
                     lw=4, alpha=0.8, zorder=1)

# # Redo legends
# R_Nax2.legend(Nlines, ['PC1','PC2','PC3','PC4','PC5'], facecolor='lightgrey',
#               fontsize='xx-large', loc=1, fancybox=True, framealpha=1)
# R_Qax2.legend(Qlines, ['PC1','PC2','PC3','PC4','PC5'], facecolor='lightgrey',
#               fontsize='xx-large', loc=1, fancybox=True, framealpha=1)

# Create new axis for the old legends
handels, labels = R_Nfig.axes[0].get_legend_handles_labels()
pc_labels = [
    'PC{}. {:03.2f}% var'.format(i+1,sorted_evals[i]*100/np.sum(sorted_evals))
    for i in range(5)
]


R_Nfig.axes[0].get_legend().remove()
R_Nax2.legend(Nlines+handels, pc_labels+labels,
              fontsize='xx-large', loc=2,
              fancybox=True, facecolor='lightgrey')

handels, labels = R_Qfig.axes[0].get_legend_handles_labels()
R_Qfig.axes[0].get_legend().remove()
R_Qax2.legend(Qlines+handels, pc_labels+labels,
              fontsize='xx-large', loc=2,
              fancybox=True, facecolor='lightgrey')


R_Nax2.set_ylabel('PC loadings', size='xx-large', color='black')
R_Qax2.set_ylabel('PC loadings', size='xx-large', color='black')

for axN, axQ in zip(R_Nfig.axes, R_Qfig.axes):
    axN.tick_params(labelsize='xx-large', length=4, color='black')
    axQ.tick_params(labelsize='xx-large', length=4, color='black')
    for labN, labQ in zip(axN.get_xticklabels(), axQ.get_xticklabels()):
        labN.set_weight('bold')
        labN.set_color('black')
        labQ.set_weight('bold')
        labQ.set_color('black')
    for labN, labQ in zip(axN.get_yticklabels(), axQ.get_yticklabels()):
        labN.set_color('black')
        labQ.set_color('black')
    axN.grid(False)
    axQ.grid(False)

R_Nfig.tight_layout()
R_Qfig.tight_layout()

R_Nfig.savefig(out_dir + base_name + '_loadings_Ri_N.png', format='png')
plt.close()
R_Qfig.savefig(out_dir + base_name + '_loadings_Ri_Q.png', format='png')
plt.close()
R_Nfig.savefig(out_dir + base_name + '_loadings_Ri_N.svg', format='svg')
plt.close()
R_Qfig.savefig(out_dir + base_name + '_loadings_Ri_Q.svg', format='svg')
plt.close()



# -------------------------------------------------------------------
# PCA PLOTS AND REGRESSION

Yvect = merged_deamid_mat.Ydata['Substrate']
Yset = set(Yvect)
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


# -------------------------------------------------------------------
### REGRESSION USING PC1 on Thermal AGE WITH BONE AND
# TAR SEEP SAMPLES
# Get known age samples
ages = merged_deamid_mat.Ydata['10C Thermal age'].astype('float')
mask = np.logical_or(Yvect=='Bone', Yvect=='Tar seep bone')
mask = np.logical_or(mask, Yvect=='Dental calculus')
ages_b = ages[mask]
known = ages_b != -1
Y = ages_b[known]
# Y = np.log10(Y+1)
X = D_imp_pca[mask, :]
X = X[known, :]
X = np.hstack((np.ones((X.shape[0],1)), X))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,1],Y)
# plt.show()
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,2],Y)
# plt.show()
plt.close()
# reg = linear_model.LinearRegression()
# reg.fit(X, Y)
#
# print(reg.score(X,Y))

# -------------------------------------------------------------------
# COMBO PCA PLOT


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
ages[known] = np.log10(ages[known]+1)

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

plt.savefig(out_dir + base_name + '_PCA_comb.png')
plt.close()

# -------------------------------------------------------------------
## PLOT BY AGE ONLY BONE AND TAR SEEP
print('Plotting bone samples')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5), dpi=300)
fig1 = plt.figure( figsize=(15,7.5))
ax1 = fig1.add_subplot(111)

bone_samples = ['Bone', 'Tar seep bone']
markers =  ['.', '^']
ages = merged_deamid_mat.Ydata['Age'].astype('float')
ages_b = ages[np.logical_or(Yvect=='Bone', Yvect=='Tar seep bone')]
known = ages_b != -1
# Logscale ages
ages_b[known] = np.log10(ages_b[known]+1)
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
    ages_s[known_s] = np.log10(ages_s[known_s]+1)
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
    ax1.scatter(ages_s[known_s], D_s[known_s,0], s=80,
                marker=m, alpha=0.9, label=s)



axes[1].legend(fontsize = 'x-large')
ax1.legend(fontsize='x-large')
# bbox_to_anchor=(1.13, 1.17)
htcbar_ax = fig.add_axes([0.3, 0.93, 0.4, 0.015])
htcbar = plt.colorbar(htsm, cax=htcbar_ax, orientation="horizontal")
htcbar.ax.set_title(r"$\log10(YBP+1)$", size='x-large')
htcbar.ax.tick_params(labelsize='x-large', length=4)

axes[0].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')
ax1.set_ylabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
ax1.set_xlabel('log10(YBP+1)', size='x-large')
for ax in axes:
    ax.tick_params(labelsize='x-large', length=4)
# Revert y axes in pc1 vs age
ax1.set_ylim(ax1.get_ylim()[::-1])

axes[0].grid(False)
axes[1].grid(False)
ax1.grid(False)

fig.savefig(out_dir + base_name + '_PCA_bone_tarseep.png', format='png')
fig1.savefig(out_dir + base_name + '_PC1_Age_bone_tarseep.png', format='png')

plt.close()

# -------------------------------------------------------------------
## PLOT BY THERMAL AGE ONLY BONE AND TAR SEEP
print('Plotting bone samples by thermal ages')

# PC 1, 2 and 3 by thermal age
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5), dpi=300)
# PC vs thermal age
fig1 = plt.figure( figsize=(15,7.5))
ax1 = fig1.add_subplot(111)

bone_samples = ['Bone', 'Tar seep bone']
markers =  ['.', '^']
ages = merged_deamid_mat.Ydata['10C Thermal age'].astype('float')
ages_b = ages[np.logical_or(Yvect=='Bone', Yvect=='Tar seep bone')]
known = ages_b != -1
# Logscale ages
ages_b[known] = np.log10(ages_b[known]+1)
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
    ages_s[known_s] = np.log10(ages_s[known_s]+1)
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
    ax1.scatter(ages_s[known_s], D_s[known_s,0], s=80,
                marker=m, alpha=0.9, label=s)



axes[1].legend(fontsize = 'x-large')
ax1.legend(fontsize='x-large')
# bbox_to_anchor=(1.13, 1.17)
htcbar_ax = fig.add_axes([0.3, 0.93, 0.4, 0.015])
htcbar = plt.colorbar(htsm, cax=htcbar_ax, orientation="horizontal")
htcbar.ax.set_title(r"$\log(10C Thermal Age+1)$", size='x-large')
htcbar.ax.tick_params(labelsize='x-large', length=4)

axes[0].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[0].set_ylabel('PC2. {:03.2f}% var'.format(sorted_evals[1]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_xlabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
axes[1].set_ylabel('PC3. {:03.2f}% var'.format(sorted_evals[2]*100/np.sum(sorted_evals)),
               size='x-large')
ax1.set_ylabel('PC1. {:03.2f}% var'.format(sorted_evals[0]*100/np.sum(sorted_evals)),
               size='x-large')
ax1.set_xlabel('log10(10C Thermal Age)', size='x-large')
for ax in axes:
    ax.tick_params(labelsize='x-large', length=4)
# Revert y axes in pc1 vs age
ax1.set_ylim(ax1.get_ylim()[::-1])
axes[0].grid(False)
axes[1].grid(False)
ax1.grid(False)
fig.savefig(out_dir + base_name + '_PCA_ThermalAge_bone_tarseep.png', format='png')
fig1.savefig(out_dir + base_name + '_PC1_ThermalAge_bone_tarseep.png', format='png')

plt.close()
