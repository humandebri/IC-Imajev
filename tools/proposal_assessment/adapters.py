"""Operator-supplied JSON command adapter; no bundled classifier dependency."""
import json
import subprocess
from .token_sweep import sha

class CommandAdapter:
    def __init__(self, name, argv, timeout=60):
        if not isinstance(argv, list) or not argv or any(type(arg) is not str or not arg for arg in argv):
            raise ValueError('nonempty command argument array required')
        self.name = name
        self.argv = list(argv)
        self.timeout = timeout

    def metadata(self):
        return dict(name=self.name, adapter='command', argv=list(self.argv),
                    provenance='provided by operator; record external model revision separately')

    def predict(self, task):
        result = subprocess.run(self.argv, check=True, timeout=self.timeout, capture_output=True,
                                text=True, input=json.dumps(task, ensure_ascii=False, allow_nan=False))
        return json.loads(result.stdout)

    def close(self):
        return None
