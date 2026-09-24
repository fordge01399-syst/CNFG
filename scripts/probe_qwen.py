from pathlib import Path
import json, os
from transformers import AutoConfig, AutoTokenizer, AutoModel
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
out=Path('pretrained/qwen25_0.5b')
out.mkdir(parents=True,exist_ok=True)
config=AutoConfig.from_pretrained(MODEL_ID)
tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModel.from_pretrained(MODEL_ID, torch_dtype='auto')
model.save_pretrained(out)
tok.save_pretrained(out)
meta={'model_id':MODEL_ID,'hidden_size':config.hidden_size,'num_hidden_layers':config.num_hidden_layers,'num_attention_heads':config.num_attention_heads,'vocab_size':config.vocab_size,'parameters':sum(p.numel() for p in model.parameters())}
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
