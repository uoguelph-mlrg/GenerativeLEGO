from DGL_DGMG import dgmg_helpers
import pickle
import torch
from DGL_DGMG.model_batch import DGMG
import json
import ast
import time
from helpers import add_node_attributes, add_edge_attributes

try:
   os.mkdir('valid_results')
except:
   pass

file = torch.load('valid.h5')
with open('valid_config.json', 'r') as f:
	config = json.load(f)
config['auto_implied_edges'] = True
config['force_valid'] = True

model = DGMG(**config)
model.load_state_dict(file['model_state_dict'])
model.eval()

eval = dgmg_helpers.LegoModelEvaluation(235, 12)
res = eval.evaluate_model(model, 200, 'valid_results')
with open('valid_results/valid_results.h5', 'wb') as f:
    pickle.dump(res, f)