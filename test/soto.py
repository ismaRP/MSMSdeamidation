from deamidation.MQreader import evidenceBatchReader
from deamidation.DSevidence import deamidationMatrix
import deamidation.reactionRates as rr
import deamidation.accFunctions as af
import numpy as np
import os


main_path = '/home/ismael/palaeoproteomics/'
datapath = main_path + 'MSMSdatasets/'

# os.system(
#     '/home/ismael/palaeoproteomics/MSMSdeamidation/sort_evidence.sh \
#     /home/ismael/palaeoproteomics/MSMSdatasets/'
# )

reader = evidenceBatchReader(
    datapath,
    prot_f = datapath + 'proteins.csv',
    samples_f = datapath + 'CollagenSamples.csv',
    ptm=['de'], aa=['QN'], tr=None, sf_exp=[])


N_properties, Q_properties = af.readHalftimes(datapath + 'N_properties.json',
                                              datapath + 'Q_properties.json')

aa_properties = N_properties.copy()
aa_properties.update(Q_properties)


mqdata = reader.readBatch()

mqdata.filter_prots(keep_names=['COL1A1'])
mqdata.filter_nonassociated_peptides(which='razor')

mqdata.createPep2Prot()
mqdata.assign_mod2prot(which='razor')
mqdata.map_positions('COL1A1', datapath+'soto_parch/20200615CSCollagenMiniDB.fasta')

for mqrun in mqdata.mqruns:
    for sample_name, sample in mqrun.samples.items():
        print('\n\n{}'.format(sample_name))
        for protid, prot in sample.prot_dict.items():
            print('{}: {}'.format(protid, prot.ptms))
