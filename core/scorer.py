import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import json
import gzip
import six
import pickle
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import math


score_scale = 5.0
min_separation = 0.25

class SCScorer():
    def __init__(self, path="data/model.ckpt-10654.as_numpy.json.gz", score_scale=5.0):
        self.vars = []
        self.score_scale = score_scale
        self.restore(path)

    def restore(self, weight_path, FP_rad=2, FP_len=1024):
        self.FP_len = FP_len; self.FP_rad = FP_rad
        self._load_vars(weight_path)

        if 'uint8' in weight_path or 'counts' in weight_path:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.FP_len,), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(mol, self.FP_rad, useChirality=True) # uitnsparsevect
                fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
                for k, v in six.iteritems(fp.GetNonzeroElements()):
                    fp_folded[k % self.FP_len] += v
                return np.array(fp_folded)
        else:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.FP_len,), dtype=np.float32)
                return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_rad, nBits=self.FP_len,
                    useChirality=True), dtype=bool)
        self.mol_to_fp = mol_to_fp

        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def apply(self, x):
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars) - 2)
            W = self.vars[i]
            b = self.vars[i+1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0) # ReLU
        x = 1 + (score_scale - 1) * 1 / (1 + np.exp(-x))
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v: print('Could not get fingerprint?')
            cur_score = 0.
        else:
            # Run
            cur_score = self.apply(fp)
            if v: print('Score: {}'.format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score)

    def _load_vars(self, weight_path):
        if weight_path.endswith('pickle'):
            with open(weight_path, 'rb') as fid:
                self.vars = pickle.load(fid)
                self.vars = [x.tolist() for x in self.vars]
        elif weight_path.endswith('json.gz'):
            with gzip.GzipFile(weight_path, 'r') as fin:    # 4. gzip
                json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
                json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]

class SAScorer():
    def __init__(self, path="data/fpscores.pkl.gz"):
        self._fscores = None
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        self.readFragmentScores(path)

    def readFragmentScores(self, name="data/fpscores.pkl.gz"):
        data = pickle.load(gzip.open(name))
        outDict = {}
        for i in data:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        self._fscores = outDict

    def numBridgeheadsAndSpiro(self, mol):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro
    
    def calculateScore(self, mol:Chem.Mol) -> float:
        """
        Calculate the SAScore for a given molecule. Scores range from 1 to 10,
        with 1 being the easiest to synthesize and 10 being the hardest.

        """
        if not mol.GetNumAtoms():
            return None
        
        sfp = self.mfpgen.GetSparseCountFingerprint(mol)

        score1 = 0.0
        nf = 0
        nze = sfp.GetNonzeroElements()
        for id, count in nze.items():
            nf += count
            score1 += self._fscores.get(id, -4) * count

        score1 /= nf

        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nBridgeheads, nSpiro = self.numBridgeheadsAndSpiro(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0

        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        score3 = 0.
        numBits = len(nze)
        if nAtoms > numBits:
            score3 = math.log(float(nAtoms) / numBits) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0

        # smooth the 10-end
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore