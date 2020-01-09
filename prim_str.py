from deamidation.MQreader import evidenceBatchReader
from deamidation.DSevidence import deamidationMatrix
import deamidation.deamidCorr as dc
import deamidation.reactionRates as rr
import deamidation.accFunctions as af
import numpy as np
import os
from scipy.stats import truncexpon
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def sort_by_lambda(properties, deamid_mat):
    lambdas = np.array([properties[trp]['lambda'] if trp in properties else -1
                        for trp in deamid_mat.trps_data['tripep']])
    idx = np.argsort(lambdas)
    lambdas = lambdas[idx]
    deamid_mat = deamid_mat.sort_tripeps(idx)
    return deamid_mat, lambdas

def create_deamid_mat(tripeps, D, t):
    trps_data = np.array(
        [tuple(['NA', 'NA', trp, 0, 0, trp]) for trp in tripeps],
        dtype=[
            ('prot_id', '<U60'),
            ('prot_name', '<U10'),
            ('tripep', 'U3'),
            ('position', 'i4'),
            ('corr_pos', 'i4'),
            ('string', '<U45')
        ]
    )
    counts = None
    rel_ints = None

    Ydata = {'times': t}
    sim_mat = deamidationMatrix(
        D=D, Ydata=Ydata, counts=counts,
        rel_ints=rel_ints, trps_data=trps_data
    )

    return sim_mat

os.chdir('/home/ismael/palaeoproteomics/')

# Read halftimes and properties
N_properties, Q_properties = af.readHalftimes('./data/N_properties.json',
                                   './data/Q_properties.json')
aa_properties = N_properties.copy()
aa_properties.update(Q_properties)

datapath = '/home/ismael/palaeoproteomics/datasets'
sf_exp = {'pompeii_cph', 'pompeii2_cph'}
out_dir = '/home/ismael/palaeoproteomics/out/'
base_name = 'logtr'

# ------------------------------------------------------------------------------
## Load data

# Read sample info
sampleInfo, header = af.readSampleInfo('./data/all_samples.tsv')
# Read proteins to filter
protsInfo = af.readProtList('./data/collagen.tsv')

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

deamid_mat = deamid_mat.merge_by_tripep()
deamid_mat = deamid_mat.filter_tripeps(deamid_mat.filter_by_pwcounts())
man_rm = ['MQG', 'HQG', 'PQL', 'DNG', 'GQH', 'GNN', 'NNG']

Qfilter = [True if trp[1]=='Q' and trp not in man_rm else False
            for trp in deamid_mat.trps_data['tripep']]
deamid_matQ = deamid_mat.filter_tripeps(Qfilter)

Nfilter = [True if trp[1]=='N' and trp not in man_rm else False
            for trp in deamid_mat.trps_data['tripep']]
deamid_matN = deamid_mat.filter_tripeps(Nfilter)

deamid_matQ, lambdasQ = sort_by_lambda(aa_properties, deamid_matQ)
deamid_matN, lambdasN = sort_by_lambda(aa_properties, deamid_matN)
fileQ = 'all_samplesQ'
fileN = 'all_samplesN'


# ------------------------------------------------------------------------------
## Simulate data

tripepsN = np.array([
    'GNP','GNI','GNV','GNR','GNK','GNT','GND','GNA','GNS',
    'ENG','VNG','PNG','HNG','LNG','ANG','KNG'
])

tripepsQ = np.array([
    'GQP','PQP','GQD','GQR','PQE','GQA','GQT','GQM',
    'DQG','EQG','SQG','LQG','FQG','KQG','VQG','PQG','IQG','AQG'
])


kN = np.array([aa_properties[trp]['lambda']
               if trp in aa_properties else -1
               for trp in tripepsN])
kQ = np.array([aa_properties[trp]['lambda']
               if trp in aa_properties else -1
               for trp in tripepsQ])

idxN = np.argsort(kN)
idxQ = np.argsort(kQ)
tripepsN = tripepsN[idxN]
kN = kN[idxN]
tripepsQ = tripepsQ[idxQ]
kQ = kQ[idxQ]

N = 20000
TmaxN = np.array([-np.log(1-0.99)/k for k in kN])
TmaxQ = np.array([-np.log(1-0.99)/k for k in kQ])
tN = [truncexpon.rvs(2.35, 0, tm/2.35, int(N/len(tripepsN))) for tm in TmaxN]
tN = np.concatenate(tN)
tQ = [truncexpon.rvs(3, 0, tm/3, int(N/len(tripepsQ))) for tm in TmaxQ]
tQ = np.concatenate(tQ)
sim_lim = 1e-7
deamid_cutoff = 0

s = 0.2
print('Simulating with sigma = {}'.format(s))
sigmasN = np.repeat(s, len(tripepsN))
sigmasQ = np.repeat(s, len(tripepsQ))

# K, sigmas, Tmean, Tstd, Tmax, N, tol
tN_s = tN[np.random.randint(tN.shape[0], size=2000)]
DN = rr.simulate_deamidation(kN, sigmasN, tN_s, sim_lim)
sim_matN = create_deamid_mat(tripepsN, DN, tN_s)

tQ_s = tQ[np.random.randint(tQ.shape[0], size=2000)]
DQ = rr.simulate_deamidation(kQ, sigmasQ, tQ_s, sim_lim)
sim_matQ = create_deamid_mat(tripepsQ, DQ, tQ_s)



# ------------------------------------------------------------------------------
## Bootstrap

# 1. Random sample (with replacement) of data and simulation
# 2. Calc Ri
# 3. Sort and rank trps according to median of Ri
# 4. Calculate Spearman's
# 5. Two plots:
#       5.1. Boxplot Spearman's Sim vs Data for Q and N
#       5.2. Boxplot of Ranks
bootstraps = 1000 # For each level of noise, repeat expetiment
bootstrap_size = 100

spearman_values = [[],[],[],[]]
b = 0
while b < bootstraps:
    # bootstrap
    dataNinds = np.random.randint(deamid_matN.D.shape[0], size=bootstrap_size)
    dataQinds = np.random.randint(deamid_matQ.D.shape[0], size=bootstrap_size)
    simNinds = np.random.randint(DN.shape[0], size=bootstrap_size)
    simQinds = np.random.randint(DQ.shape[0], size=bootstrap_size)

    dataDN_b = deamid_matN.D[dataNinds,:]
    dataDQ_b = deamid_matQ.D[dataQinds,:]
    dataCN_b = deamid_matN.counts[dataNinds,:]
    dataCQ_b = deamid_matQ.counts[dataQinds,:]
    simDNb_b = DN[simNinds,:]
    simDQb_b = DQ[simQinds,:]

    dataRN_b = rr.calc_Ri(dataDN_b, dataCN_b, 4, num_tol=deamid_cutoff)
    dataRQ_b = rr.calc_Ri(dataDQ_b, dataCQ_b, 4, num_tol=deamid_cutoff)
    simRN_b = rr.calc_Ri(simDNb_b, None, None, num_tol=deamid_cutoff)
    simRQ_b = rr.calc_Ri(simDQb_b, None, None, num_tol=deamid_cutoff)

    data_argsortN = np.argsort([np.median(v) for v in dataRN_b])
    data_argsortQ = np.argsort([np.median(v) for v in dataRQ_b])
    sim_argsortN = np.argsort([np.median(v) for v in simRN_b])
    sim_argsortQ = np.argsort([np.median(v) for v in simRQ_b])

    data_corrN = spearmanr(data_argsortN, np.array(range(len(data_argsortN)))).correlation
    data_corrQ = spearmanr(data_argsortQ, np.array(range(len(data_argsortQ)))).correlation
    sim_corrN = spearmanr(sim_argsortN, np.array(range(len(sim_argsortN)))).correlation
    sim_corrQ = spearmanr(sim_argsortQ, np.array(range(len(sim_argsortQ)))).correlation

    spearman_values[0].append(data_corrN)
    spearman_values[1].append(data_corrQ)
    spearman_values[2].append(sim_corrN)
    spearman_values[3].append(sim_corrQ)

    b += 1


spearman_values = np.array(spearman_values)
labs = [
    'XNZ real data',
    'XQZ real data',
    'XNZ simulation',
    'XQZ simulation'
]
fig = plt.figure(figsize=(9.6, 7.2), dpi=300)
ax = fig.add_subplot(111)
ax.boxplot(spearman_values.T, labels = labs)
ax.set_ylabel('Spearman\'s '+r'$\rho$')
plt.savefig(out_dir + 'data_vs_sim/' + base_name + '_bootstrap_spearman.png')
plt.close()
#-------------------------------------------------------------------------------
## Plot histograms of deamidation for data vs simulations

## Plot histograms for all N and Q
rr.data_vs_sim_hist(
    data=np.array([np.concatenate(deamid_matN.D)]),
    sim=np.array([np.concatenate(DN)]),
    trps=['XNY'], xlabel='Deamidation',
    xrange=None,
    path=out_dir+'data_vs_sim/', base_name=base_name+'N_D'
)
rr.data_vs_sim_hist(
    data=np.array([np.concatenate(deamid_matQ.D)]),
    sim=np.array([np.concatenate(DQ)]),
    trps=['XQY'], xlabel='Deamidation',
    xrange=None,
    path=out_dir+'data_vs_sim/', base_name=base_name+'Q_D'
)

## Plot histograms for each tripeptide
rr.data_vs_sim_hist(
    data=deamid_matN.D, sim=DN,
    trps=deamid_matN.trps_data['tripep'],
    xlabel='Deamidation', xrange=None,
    path=out_dir+'data_vs_sim/', base_name=base_name+'N_D_trps'
)
rr.data_vs_sim_hist(
    data=deamid_matQ.D, sim=DQ,
    trps=deamid_matQ.trps_data['tripep'],
    xlabel='Deamidation', xrange=None,
    path=out_dir+'data_vs_sim/', base_name=base_name+'Q_D_trps'
)

# ------------------------------------------------------------------------------
## Calculate R
dataR_N = rr.calc_Ri(
    deamid_matN.D,
    deamid_matN.counts,
    low_counts=4,
    num_tol=deamid_cutoff
)

dataR_Q = rr.calc_Ri(
    deamid_matQ.D,
    deamid_matQ.counts,
    low_counts=4,
    num_tol=deamid_cutoff
)

# Calculate rho
argsortN = np.argsort([np.median(v) for v in dataR_N])
argsortQ = np.argsort([np.median(v) for v in dataR_Q])
corrN = spearmanr(argsortN, np.array(range(len(argsortN)))).correlation
corrQ = spearmanr(argsortQ, np.array(range(len(argsortQ)))).correlation
print('All samples, XNZ, rho={}'.format(corrN))
print('All samples, XQZ, rho={}'.format(corrQ))

simR_N = rr.calc_Ri(
    DN,
    None,
    None,
    num_tol=deamid_cutoff
)
simR_Q = rr.calc_Ri(
    DQ,
    None,
    None,
    num_tol=deamid_cutoff
)

# Histograms
rr.data_vs_sim_hist(
    data=dataR_N, sim=simR_N,
    trps=deamid_matN.trps_data['tripep'],
    xlabel='R', xrange=[0, 50],
    path=out_dir+'data_vs_sim/', base_name=base_name+'N_R_trps'
)
rr.data_vs_sim_hist(
    data=dataR_Q, sim=simR_Q,
    trps=deamid_matQ.trps_data['tripep'],
    xlabel='R', xrange=[0, 50],
    path=out_dir+'data_vs_sim/', base_name=base_name+'Q_R_trps'
)

# QQplots
rr.R_QQplot(simR_N, dataR_N, ['Simulated', 'Data'],
            deamid_matN.trps_data['tripep'], out_dir+'by_tripep/',
            aspect='num')
rr.R_QQplot(simR_Q, dataR_Q, ['Simulated', 'Data'],
            deamid_matQ.trps_data['tripep'], out_dir+'by_tripep/',
            aspect='num')

## Aggregate R_i values
aggr_simR_N = [[val for v in simR_N for val in v]]
aggr_simR_Q = [[val for v in simR_Q for val in v]]
aggr_dataR_N = [[val for v in dataR_N for val in v]]
aggr_dataR_Q = [[val for v in dataR_Q for val in v]]

rr.R_QQplot(aggr_simR_N, aggr_simR_Q, ['Simulated','Simulated'],
            ['XNZ', 'XQZ'], out_dir+'by_tripep/', base_name+'simXNZ_simXQZ',
            aspect='equal')
rr.R_QQplot(aggr_dataR_N, aggr_dataR_Q, ['Data', 'Data'],
            ['XNZ', 'XQZ'], out_dir+'by_tripep/', base_name+'dataXNZ_dataXQZ',
            aspect='equal')

rr.R_QQplot(aggr_simR_N, aggr_dataR_N, ['Simulated', 'Data'],
            ['XNZ', 'XNZ'], out_dir+'by_tripep/', base_name+'simXNZ_dataXNZ',
            aspect='num')
rr.R_QQplot(aggr_simR_Q, aggr_dataR_Q, ['Simulated', 'Data'],
            ['XQZ', 'XQZ'], out_dir+'by_tripep/', base_name+'simXQZ_dataXQZ',
            aspect='num')

# ------------------------------------------------------------------------------
## Calculate ranks
data_rankN = rr.calc_rankj_values(
    deamid_matN.D,
    deamid_matN.counts,
    low_counts=4,
    num_tol=deamid_cutoff
)
data_rankQ = rr.calc_rankj_values(
    deamid_matQ.D,
    deamid_matQ.counts,
    low_counts=4,
    num_tol=deamid_cutoff
)

sim_rankN = rr.calc_rankj_values(
    DN,
    None,
    None,
    num_tol=deamid_cutoff
)
sim_rankQ = rr.calc_rankj_values(
    DQ,
    None,
    None,
    num_tol=deamid_cutoff
)

# Histograms
rr.data_vs_sim_hist(
    data=data_rankN, sim=sim_rankN,
    trps=deamid_matN.trps_data['tripep'],
    xlabel='Rank', xrange=None,
    path=out_dir+'data_vs_sim/', base_name=base_name+'N_rank_trps'
)
rr.data_vs_sim_hist(
    data=data_rankQ, sim=sim_rankQ,
    trps=deamid_matQ.trps_data['tripep'],
    xlabel='Rank', xrange=None,
    path=out_dir+'data_vs_sim/', base_name=base_name+'Q_rank_trps'
)

# QQplots
# rr.R_QQplot(sim_rankN, data_rankN, deamid_matN.trps_data,
#             out_dir+'by_tripep/rank', aspect='equal')
# rr.R_QQplot(sim_rankQ, data_rankQ, deamid_matQ.trps_data,
#             out_dir+'by_tripep/rank', aspect='equal')


# ------------------------------------------------------------------------------
## Data plots

rr.Rlambda_distr(dataR_N, deamid_matN.trps_data,
                 sort_by=None, log=True,
                 path=out_dir, base_name=base_name+fileN)

rr.Rlambda_distr(dataR_Q, deamid_matQ.trps_data,
                 sort_by=None, log=True,
                 path=out_dir, base_name=base_name+fileQ)

# Multiscatter plots
dc.multiscatter(deamid_matQ, hts=lambdasQ, t='norm',
                path=out_dir, base_name=base_name+fileQ, low_counts=4,
                fontsize=10, fontpos=[0.10,0.35], reg=False)
dc.multiscatter(deamid_matN, hts=lambdasN, t='norm',
                path=out_dir, base_name=base_name+fileN, low_counts=4,
                fontsize=10, fontpos=[0.10,0.35], reg=False)
#
# # Calculate correlation
# corr_matQ = dc.correlation(deamid_matQ.D, deamid_matQ.trps_data,
#                            out_dir, base_name+fileQ)
# corr_matN = dc.correlation(deamid_matN.D, deamid_matN.trps_data,
#                            out_dir, base_name+fileN)
#
# grQ = dc.show_graph_with_labels(corr_matQ, 0.8, deamid_matQ.trps_data['tripep'],
#                                 aa_properties, 'polarity_graham',
#                                 out_dir, base_name+fileQ)
# grN = dc.show_graph_with_labels(corr_matN, 0.7, deamid_matN.trps_data['tripep'],
#                                 aa_properties, 'polarity_graham',
#                                 out_dir, base_name+fileN)


# By substrate
substrates = ['Bone', 'Dental calculus', 'Gelatine', 'Skin', 'Parchment']
for sub in substrates:
    if sub == 'Bone':
        b_mask = deamid_mat.Ydata['Substrate'] == sub
        ts_mask = deamid_mat.Ydata['Substrate'] == 'Tar seep bone'
        mask = np.logical_or(b_mask, ts_mask)
    else:
        mask = deamid_mat.Ydata['Substrate'] == sub
    sub_deamid_mat = deamid_mat.filter_samples(mask)
    print(sub)
    Qfilter = [True if trp[1]=='Q' and trp not in man_rm else False
                for trp in sub_deamid_mat.trps_data['tripep']]
    sub_deamid_matQ = sub_deamid_mat.filter_tripeps(Qfilter)

    Nfilter = [True if trp[1]=='N' and trp not in man_rm else False
                for trp in sub_deamid_mat.trps_data['tripep']]
    sub_deamid_matN = sub_deamid_mat.filter_tripeps(Nfilter)

    sub_deamid_matQ, lambdasQ = sort_by_lambda(aa_properties, sub_deamid_matQ)
    sub_deamid_matN, lambdasN = sort_by_lambda(aa_properties, sub_deamid_matN)
    sub_Ri_Q = rr.calc_Ri(
        sub_deamid_matQ.D,
        sub_deamid_matQ.counts,
        low_counts=4,
        num_tol=deamid_cutoff
    )
    sub_Ri_N = rr.calc_Ri(
        sub_deamid_matN.D,
        sub_deamid_matN.counts,
        low_counts=4,
        num_tol=deamid_cutoff
    )

    # Calculate rho
    argsortN = np.argsort([np.median(v) for v in sub_Ri_N])
    argsortQ = np.argsort([np.median(v) for v in sub_Ri_Q])
    corrN = spearmanr(argsortN, np.array(range(len(argsortN)))).correlation
    corrQ = spearmanr(argsortQ, np.array(range(len(argsortQ)))).correlation
    print('{} samples, XNZ, rho={}'.format(sub, corrN))
    print('{} samples, XQZ, rho={}'.format(sub, corrQ))

    fileQ = sub + '_Q'
    fileN = sub + '_N'
    # Calculate relative ratio
    rr.Rlambda_distr(sub_Ri_Q, sub_deamid_matQ.trps_data,
                     sort_by=None, log=True,
                     path=out_dir, base_name=base_name+fileQ)
    rr.Rlambda_distr(sub_Ri_N, sub_deamid_matN.trps_data,
                     sort_by=None, log=True,
                     path=out_dir, base_name=base_name+fileN)
