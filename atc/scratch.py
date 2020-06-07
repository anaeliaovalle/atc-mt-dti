import pubchempy as pcp
import pdb

pdb.set_trace()
res = pcp.get_compounds('COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl', namespace='smiles')


res_met = pcp.get_compounds('ibuprofen', namespace='name')[0]