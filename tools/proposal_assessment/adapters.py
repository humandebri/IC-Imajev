"""Pluggable offline classifiers: local Laya or a JSON stdin/stdout command."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class CommandAdapter:
    """One JSON task on stdin, one JSON response on stdout; no shell evaluation."""
    def __init__(self, name, argv, timeout=60):
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
            raise ValueError('command must be a nonempty JSON array of strings')
        self.name, self.argv, self.timeout = name, argv, timeout

    def metadata(self):
        return {'name': self.name, 'adapter': 'command', 'argv': self.argv,
                'provenance': 'provided by operator; record external model revision separately'}

    def predict(self, task):
        process = subprocess.run(self.argv, input=json.dumps(task, ensure_ascii=False, allow_nan=False),
                                 capture_output=True, text=True, timeout=self.timeout, check=True)
        return json.loads(process.stdout)

    def close(self):
        pass


class LayaAdapter:
    """Existing reviewed local upstream DecisionModel; never fetches remote code."""
    def __init__(self, name, source, upstream, max_tokens=128):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
        self.torch = torch
        source, upstream = Path(source), Path(upstream)
        config = json.loads((source / 'rl_agent_config.json').read_text())
        encoder_config = AutoConfig.from_pretrained(source / 'encoder', local_files_only=True)
        if encoder_config.model_type != 'modernbert':
            raise ValueError('Laya adapter supports ModernBERT checkpoints; use command adapter for other architectures')
        if not 1 <= max_tokens <= min(config['max_len'], encoder_config.max_position_embeddings):
            raise ValueError('max_tokens exceeds checkpoint context')
        self.max_tokens = max_tokens
        files = ('model.safetensors', 'encoder/config.json', 'rl_agent_config.json',
                 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json')
        self._metadata = {'name': name, 'adapter': 'laya', 'mode': 'CPU F32 eager',
                          'max_tokens': max_tokens, 'torch_version': torch.__version__,
                          'source_sha256': {f: sha(source / f) for f in files},
                          'upstream_sha256': sha(upstream), 'calibration_applied': False}
        provenance = source / 'comparison-provenance.json'
        if provenance.exists():
            self._metadata['checkpoint_provenance'] = json.loads(provenance.read_text())
        spec = importlib.util.spec_from_file_location('proposal_laya_common', upstream)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        torch.set_num_threads(4)
        with torch.device('meta'):
            encoder = AutoModel.from_config(encoder_config, attn_implementation='eager')
            model = common.DecisionModel(encoder, head_layers=config['head_layers'],
                                         n_act=len(config['act_costs']) + 1)
        model.load_state_dict(load_file(source / 'model.safetensors'), assign=True, strict=True)
        self.model = model.float().eval()
        for module_name, module in list(self.model.named_modules()):
            if isinstance(module, ModernBertRotaryEmbedding):
                parent, _, attr = module_name.rpartition('.')
                setattr(self.model.get_submodule(parent), attr, ModernBertRotaryEmbedding(encoder_config, device='cpu'))
        if any(b.is_meta for b in self.model.buffers()):
            raise ValueError('uninitialized model buffer')
        self.tokenizer = AutoTokenizer.from_pretrained(source / 'tokenizer', local_files_only=True)

    def metadata(self):
        return self._metadata

    def predict(self, task):
        tok, torch = self.tokenizer, self.torch
        enc = lambda text: tok(text, add_special_tokens=False)['input_ids']
        # Refuse control-token literals, rather than silently changing source text.
        texts = [task['question'], task['state'], *task['options']]
        if any(special in text for special in tok.all_special_tokens for text in texts):
            raise ValueError('text contains tokenizer control-token literal')
        ids = [tok.cls_token_id, *enc('choice question: ' + task['question']), tok.sep_token_id]
        markers = []
        for option in task['options']:
            markers.append(len(ids))
            rendered = option + ': ' + task['option_descriptions'][option]
            ids.extend([tok.mask_token_id, *enc(' ' + rendered)])
        ids.extend([tok.sep_token_id, *enc(task['state']), tok.sep_token_id])
        if len(ids) > self.max_tokens:
            raise ValueError(f'input has {len(ids)} tokens, exceeds {self.max_tokens}; no truncation')
        with torch.inference_mode():
            logits, _ = self.model(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long),
                                   torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool),
                                   torch.tensor([0]))
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite logits')
        values = logits[0].tolist()
        return {'label': task['options'][max(range(len(values)), key=values.__getitem__)],
                'logits': values, 'input_tokens': len(ids), 'evidence': []}

    def close(self):
        import gc
        del self.model
        gc.collect()
