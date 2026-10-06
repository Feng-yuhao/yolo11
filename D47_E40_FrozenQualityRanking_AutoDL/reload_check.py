"""Isolated-process standard state-dict reload used by unit_checks."""
import sys
import torch
from ranking import Ranker

saved=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
model=Ranker(saved['variant'],saved['mean'].numpy(),saved['std'].numpy())
model.load_state_dict(saved['state'],strict=True)
model.eval()
torch.testing.assert_close(model(saved['input']),saved['output'],rtol=2e-5,atol=2e-7)
print('D47 fresh CPU process ranker reload passed')
