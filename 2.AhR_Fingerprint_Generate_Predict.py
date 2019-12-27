"""
ex.   /diurons.csv
CASRN,SMILES
Diuron,CN(C)C(=O)NC1=CC(Cl)=C(Cl)C=C1
34-Dichloroaniline,NC1=CC(Cl)=C(Cl)C=C1
34-Dichlorophenylurea,NC(=O)NC1=CC=C(Cl)C(Cl)=C1
1-Aziridinecarboxamide,ClC1=C(Cl)C=C(NC(=O)N2CC2)C=C1"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy


def SmiData2FP(path, output):
    smiles_file = pd.read_csv(path)
    ID_list = []
    mols = []
    for index, smiles in zip(smiles_file['CASRN'], smiles_file['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)
        ID_list.append(index)
    feature_list = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
    np_fps = []
    [np_fps.append(numpy.asarray(i)) for i in feature_list]
    ID_feature_df = pd.DataFrame(np_fps, ID_list)
    vec_name = ['feature_{0}'.format(i) for i in range(0, 2048)]
    ID_feature_df.columns = vec_name
    ID_feature_df.index.name = 'CASRN'
    file = ID_feature_df.dropna(axis=0, how='any')
    file.to_csv(output)  # 保存结果到文件


if __name__ == '__main__':
    path = r'pesticides.csv'
    output = r'pesticides_FP.csv'
    SmiData2FP(path, output)
    print('------------------Done------------------')
