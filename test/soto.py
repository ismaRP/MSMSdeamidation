from deamidation.MQreader import evidenceBatchReader
from deamidation.DSevidence import deamidationMatrix
import deamidation.reactionRates as rr
import deamidation.accFunctions as af
import numpy as np
import os


main_path = '/home/ismael/palaeoproteomics/'
datapath = main_path + 'MSMSdatasets/'

os.system(
    '/home/ismael/palaeoproteomics/MSMSdeamidation/sort_evidence.sh \
    /home/ismael/palaeoproteomics/MSMSdatasets/'
)

reader = evidenceBatchReader(
    datapath,
    prot_f = datapath + 'proteins.csv',
    samples_f = datapath + 'CollagenSamples.csv',
    ptm=['de'], aa=['QN'], tr='log', int_threshold=150, sf_exp=[])


N_properties, Q_properties = af.readHalftimes(datapath + 'N_properties.json',
                                              datapath + 'Q_properties.json')

aa_properties = N_properties.copy()
aa_properties.update(Q_properties)


mqdata = reader.readBatch()

mqdata.plot_intensities(main_path+'MSMSout/int_vs_len.png')

mqdata.filter_prots(keep_names=['COL1A1'])
mqdata.filter_nonassociated_peptides(which='razor')

mqdata.createPep2Prot()
mqdata.assign_mod2prot(which='razor')

# mqdata.map_positions('COL1A1', datapath+'soto_parch/20200615CSCollagenMiniDB.fasta')
mqdata.map_positions('COL1A1', alignment='COL1A1.aln')



anndata = mqdata.create_pxm_matrix(prot_name='COL1A1')
anndata = anndata[np.all(np.isnan(anndata.X), axis=1),:]
anndata = anndata.groupby_sample()
