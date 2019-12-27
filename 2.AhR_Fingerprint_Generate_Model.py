import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy


def ToxFile2SmiData(smiles_input, smiles_output):
	"""format in
	smiles \t name \t activity"""
	compound_ID = []
	smiles = []
	activity = []
	with open(smiles_input) as text:
		for line in text.readlines():
			str = line.strip().split()
			compound_ID.append(str[1])
			smiles.append(str[0])
			activity.append(str[2])
	data = {'compound_ID': compound_ID,
			'smiles': smiles,
			'activity': activity}
	smiles_file = pd.DataFrame(data)
	smiles_file.to_csv(smiles_output)
	return


def SmiData2FP(smiles_output, fingerprints_output):
	smiles_file = pd.read_csv(smiles_output)
	ID_list = []
	activ_list = []
	mols = []
	for index, smiles, activ in zip(smiles_file['compound_ID'],smiles_file['smiles'],smiles_file['activity']):
		mol = Chem.MolFromSmiles(smiles)
		mols.append(mol)
		ID_list.append(index)
		activ_list.append(activ)
	feature_list = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
	np_fps = []
	[np_fps.append(numpy.asarray(i)) for i in feature_list]
	ID_feature_df = pd.DataFrame(np_fps, ID_list)
	vec_name = ['feature_{0}'.format(i) for i in range(0,2048)]
	ID_feature_df.columns = vec_name
	ID_feature_df.index.name = 'compound'
	ID_feature_df['activity'] = activ_list
	file = ID_feature_df.dropna(axis=0, how='any')
	file.to_csv(fingerprints_output)
	return


if __name__ == '__main__':
	smiles_input = r'1.ahr_tox.smiles'
	smiles_output = r'AhR_Smiles_File.csv'
	FP_output = r'Model_FP.csv'

	ToxFile2SmiData(smiles_input, smiles_output)
	SmiData2FP(smiles_output, FP_output)
	print('------------------Done------------------')
