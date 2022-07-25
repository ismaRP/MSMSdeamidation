import numpy as np
from deamidation.MQreader import EvidenceBatchReader
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

main_path = '/home/ismael/palaeoproteomics/'
datapath = main_path + 'MSMSdatasets/'
mspath = datapath + 'mq/'
out_dir = datapath + '/out/'


samples_df = pd.read_csv(datapath + 'all_samples.tsv',
                         index_col=0,
                         sep='\t')
datasets = {'soto', 'gelatine fish'}
samples_df = samples_df[~samples_df.Dataset.isin(datasets)]
prots_df = pd.read_csv(datapath + 'proteins.csv',
                       index_col=0,
                       sep=',')
sf_exp = {'pompeii_cph', 'pompeii2_cph'}
reader = EvidenceBatchReader(
    mspath,
    prot_f=prots_df,
    samples_f=samples_df,
    fasta_f='',
    ptm=['de', 'hy', ''], aa=['QN', 'P', 'C'], tr='log', int_threshold=150, sf_exp=sf_exp)

print('reading...')
mqdata = reader.readBatch()

print('filtering for COL1A1 and COL1A2...')
mqdata.filter_prots(keep_names=['COL1A1', 'COL1A2'])
mqdata.filter_nonassociated_peptides(which='razor')

print('assigning PTMs to proteins...')
mqdata.createPep2Prot()
mqdata.assign_mod2prot(which='razor')

# pos_map = mqdata.map_positions(['COL1A1', 'COL1A2'], alnformat='clustal', aamod='PC', return_map=True)


def calc_cor(hdvector):
    hd = hdvector[:, 0:2]
    weights = hdvector[:, 2]
    if hd.shape[0] == 1:
        correlation = 0
    else:
        cov_m = np.cov(hd, rowvar=False, aweights=weights)
        if cov_m[0, 0] == 0 or cov_m[1, 1] == 0:
            correlation = 0
        else:
            correlation = cov_m[1, 0] / np.sqrt(cov_m[0, 0] * cov_m[1, 1])
    return correlation


hd_tripep_global = {}
hd_pept_global = {}
for mqrun in mqdata.mqruns:
    for sample_name, sample in mqrun.samples.items():
        hd_pept = {}
        hd_tripep = {}
        for pept in sample.pept_dict.values():
            if 'Q' not in pept.sequence or 'N' not in pept.sequence:
                continue
            if 'P' not in pept.sequence:
                continue
            if pept.sequence not in hd_pept:
                hd_pept[pept.sequence] = []
            pep_trps = []
            intensity = pept.intensity
            # Deamidated in 0
            # Hydroxylated in 1
            vector = [0, 0, intensity]
            for ptm in pept.ptms:
                pos = ptm[2]
                trp = pept.sequence[pos-2:pos+1]
                if len(trp) < 3 or (trp[0] != 'G' and trp[2] != 'G'):
                    continue
                if ptm[0] == 'Q' or ptm[0] == 'N':
                    pep_trps.append(trp)
                if ptm[1] == 'Q(de)' or ptm[1] == 'N(de)':
                    vector[0] = 1
                elif ptm[1] == 'P(hy)':
                    vector[1] = 1
            hd_pept[pept.sequence].append(vector)
            for trp in pep_trps:
                if trp not in hd_tripep:
                    hd_tripep[trp] = []
                hd_tripep[trp].append(vector)

        for pept, hdvector in hd_pept.items():
            hdvector = np.array(hdvector)
            correlation = calc_cor(hdvector)
            hd_pept[pept] = correlation
        hd_pept_global[sample_name] = hd_pept
        for trp, hdvector in hd_tripep.items():
            hdvector = np.array(hdvector)
            correlation = calc_cor(hdvector)
            hd_tripep[trp] = correlation
        hd_tripep_global[sample_name] = hd_tripep

hd_tripep_global = pd.DataFrame(hd_tripep_global)
hd_tripep_global['tripeptide'] = hd_tripep_global.index
print(hd_tripep_global)
hd_tripep_global = pd.melt(hd_tripep_global, id_vars=['tripeptide'], var_name='sample')
print(hd_tripep_global)

fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.boxplot(data=hd_tripep_global, x='tripeptide', y='value',
                 ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.savefig(out_dir + 'hyd_deam_corr.png')

hd_pept_global = pd.DataFrame(hd_pept_global)
print(hd_pept_global)

