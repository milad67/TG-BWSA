"""Collect the active Python environment without running training or inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile


def redact(text):
    text = re.sub(r'(https?://)[^/\s]+@', r'\1<credentials-redacted>@', text)
    return re.sub(r'(?i)([?&](?:token|access_token|key|password)=)[^&\s]+', r'\1<redacted>', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='server_environment')
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'collected_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'python_executable': sys.executable,
                'platform': platform.platform(), 'commands': []}
    commands = [('requirements_pip.txt', [sys.executable, '-m', 'pip', 'freeze']),
                ('pytorch_environment.txt', [sys.executable, '-m', 'torch.utils.collect_env'])]
    for name, command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            (out/name).write_text(redact(result.stdout), encoding='utf-8')
            if result.stderr:
                (out/(name+'.stderr.txt')).write_text(redact(result.stderr), encoding='utf-8')
            metadata['commands'].append({'file': name, 'returncode': result.returncode})
        except (OSError, subprocess.TimeoutExpired) as exc:
            metadata['commands'].append({'file': name, 'error': str(exc)})
    (out/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    archive = out.with_name(out.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            z.write(path, path.name)
    print(f'Created {archive}')
    if any(c.get('returncode', 1) != 0 for c in metadata['commands']):
        print('Some commands could not complete; their diagnostics are included in the archive.')


if __name__ == '__main__':
    main()
