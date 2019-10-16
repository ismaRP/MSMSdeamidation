from deamidation.DSevidence import deamidationMatrix
import deamidation.deamidCorr as dc
import deamidation.reactionRates as rr
import deamidation.accFunctions as af
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import truncexpon
from scipy.stats import spearmanr
import os
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-dark-palette')

out_dir = '/home/ismael/palaeoproteomics/out/'


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

def sigma_spearman_plot(corrs, sigmas, aa, out_dir, base_name):
    fig = plt.figure(figsize=(9.6, 7.2), dpi=300)

    for i in range(len(corrs)):

        ax = fig.add_subplot(2, 1, i+1)
        ax.boxplot(corrs[i], positions=sigmas, widths=0.03)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                           size='medium')
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(r'$\sigma$', size='xx-large')
        ax.set_ylabel('Spearman\'s '+r'$\rho$', size='xx-large')
        ax.set_title('X'+aa[i]+'Z')
        if i == 0:
            ax.xaxis.set_visible(False)
    plt.savefig(out_dir + base_name + '_sigmaVSrho.png')
    plt.close()

N_properties, Q_properties = af.readHalftimes('data/N_properties.json',
                                   'data/Q_properties.json')
aa_properties = N_properties.copy()
aa_properties.update(Q_properties)


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
tN = [truncexpon.rvs(2.5, 0, tm/2.5, int(N/len(tripepsN))) for tm in TmaxN]
tN = np.concatenate(tN)
tQ = [truncexpon.rvs(2.5, 0, tm/2.5, int(N/len(tripepsQ))) for tm in TmaxQ]
tQ = np.concatenate(tQ)
sim_lim = 1e-7
deamid_cutoff = 0

# different levels of noise
sigmas = [1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.005]
bootstraps = 500 # For each level of noise, repeat expetiment
bootstrap_size = 100
corrsN_1 = []; corrsQ_1 = [] # sigmas x bootstraps
corrsN_2 = []; corrsQ_2 = []
corrsN_3 = []; corrsQ_3 = []

for s in sigmas:
    print('Simulating with sigma = {}'.format(s))
    sigmasN = np.repeat(s, len(tripepsN))
    sigmasQ = np.repeat(s, len(tripepsQ))

    corrsN_1_b = []; corrsQ_1_b = []
    corrsN_2_b = []; corrsQ_2_b = []
    corrsN_3_b = []; corrsQ_3_b = []
    b = 0 # Bootstrap counter
    while b < bootstraps:
        # Samples times
        tN_b = tN[np.random.randint(tN.shape[0], size=bootstrap_size)]
        tQ_b = tQ[np.random.randint(tQ.shape[0], size=bootstrap_size)]

        DN_b = rr.simulate_deamidation(kN, sigmasN, tN_b, sim_lim)
        DQ_b = rr.simulate_deamidation(kQ, sigmasQ, tQ_b, sim_lim)

        RN = rr.calc_Ri(DN_b, num_tol=deamid_cutoff)
        RQ = rr.calc_Ri(DQ_b, num_tol=deamid_cutoff)

        argsortN = np.argsort([np.median(v) for v in RN])
        argsortQ = np.argsort([np.median(v) for v in RQ])

        simple_rankN = rr.calc_rankRj(DN_b, num_tol=deamid_cutoff)
        simple_rankQ = rr.calc_rankRj(DQ_b, num_tol=deamid_cutoff)

        values_rankN = rr.calc_rankj_values(DN_b, num_tol=deamid_cutoff)
        values_rankQ = rr.calc_rankj_values(DQ_b, num_tol=deamid_cutoff)
        sort_rankN = np.argsort([np.median(v) for v in RN])
        sort_rankQ = np.argsort([np.median(v) for v in RQ])



        # Use spearman correlation as "sortness" measure
        corrN = spearmanr(argsortN, np.array(range(len(argsortN)))).correlation
        corrQ = spearmanr(argsortQ, np.array(range(len(argsortQ)))).correlation
        corrsN_1_b.append(corrN)
        corrsQ_1_b.append(corrQ)

        corrN = spearmanr(simple_rankN, np.array(range(len(simple_rankN)))).correlation
        corrQ = spearmanr(simple_rankQ, np.array(range(len(simple_rankQ)))).correlation
        corrsN_2_b.append(corrN)
        corrsQ_2_b.append(corrQ)

        corrN = spearmanr(sort_rankN, np.array(range(len(sort_rankN)))).correlation
        corrQ = spearmanr(sort_rankQ, np.array(range(len(sort_rankQ)))).correlation
        corrsN_3_b.append(corrN)
        corrsQ_3_b.append(corrQ)



        b += 1
    corrsN_1.append(corrsN_1_b); corrsQ_1.append(corrsQ_1_b)
    corrsN_2.append(corrsN_2_b); corrsQ_2.append(corrsQ_2_b)
    corrsN_3.append(corrsN_3_b); corrsQ_3.append(corrsQ_3_b)

corrs_1 = [corrsN_1, corrsQ_1]
corrs_2 = [corrsN_2, corrsQ_2]
corrs_3 = [corrsN_3, corrsQ_3]
aa = ['N', 'Q']

# sigma_spearman_plot(corrs_1, sigmas, aa, out_dir, 'Ri')
# sigma_spearman_plot(corrs_2, sigmas, aa, out_dir, 'simple_rank')
# sigma_spearman_plot(corrs_3, sigmas, aa, out_dir, 'values_rank')


print('Done')

####### Simulate data at 0.15 noise
s = 0.15
print('Simulating with sigma = {}'.format(s))
sigmasN = np.repeat(s, len(tripepsN))
sigmasQ = np.repeat(s, len(tripepsQ))

# K, sigmas, Tmean, Tstd, Tmax, N, tol
tN_s = tN[np.random.randint(tN.shape[0], size=2000)]
DN = rr.simulate_deamidation(kN, sigmasN, tN_s, sim_lim)
sim_matN = create_deamid_mat(tripepsN, DN, tN_s)
RN = rr.calc_Ri(DN, num_tol=deamid_cutoff)
rankN = rr.calc_rankj_values(DN, num_tol=0)

tQ_s = tQ[np.random.randint(tQ.shape[0], size=2000)]
DQ = rr.simulate_deamidation(kQ, sigmasQ, tQ_s, sim_lim)
sim_matQ = create_deamid_mat(tripepsQ, DQ, tQ_s)
RQ = rr.calc_Ri(DQ, num_tol=deamid_cutoff)
rankQ = rr.calc_rankj_values(DQ, num_tol=0)

rr.Rlambda_distr(rankN, trps_data=sim_matN.trps_data,
                 sort_by='mean',
                 path=out_dir, base_name='simN_1_rank',
                 log=False)

rr.Rlambda_distr(rankQ, trps_data=sim_matQ.trps_data,
                 sort_by='mean',
                 path=out_dir, base_name='simQ_1_rank',
                 log=False)


print('Creating multiscatter...')
dc.multiscatter(
    sim_matN, hts=kN,
    t='norm', path=out_dir, base_name='simN_1',
    low_counts=None, fontsize=10, fontpos=[0.10,0.35],
    reg=False
)

dc.multiscatter(
    sim_matQ, hts=kQ,
    t='norm', path=out_dir, base_name='simQ_1',
    low_counts=None, fontsize=10, fontpos=[0.10,0.35],
    reg=False
)

print('Done')

print('Creating violin plots...')
print('For N...')
rr.Rlambda_distr(RN, trps_data=sim_matN.trps_data,
                 sort_by='median',
                 path=out_dir, base_name='simN_1',
                 log=False)
print('For Q...')
rr.Rlambda_distr(RQ, trps_data=sim_matQ.trps_data,
                 sort_by='median',
                 path=out_dir, base_name='simQ_1',
                 log=False)
print('Done')
