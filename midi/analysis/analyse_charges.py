import pickle
from tqdm import tqdm
from rdkit import Chem


file = '/home/vignac/MoleculeDiffusion/data/geom/rdkit_folder/rdkit_mols.pickle'

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_smiles_and_save(path):
    all_smiles = [smiles for (smiles, _) in tqdm(load_pickle(path))]
    with open('all_smiles.txt', 'w') as f:
        for smiles in all_smiles:
            f.write(smiles + '\n')
    return


# extract_smiles_and_save(file)



smiles_file = '/Users/clementvignac/src/github_cvignac/MoleculeDiffusion/data/geom/all_smiles.txt'



