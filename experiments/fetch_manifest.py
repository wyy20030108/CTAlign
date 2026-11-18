"""Fetch only the versioned CT-RATE report tables and ReX split metadata."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError


FILES = {
    'ct_rate': (
        'ibrahimhamamci/CT-RATE',
        '546640db5d4674f6e6b36d40d2cfcd57c8b39e21',
        ('dataset/radiology_text_reports/train_reports.csv',
         'dataset/radiology_text_reports/validation_reports.csv'),
    ),
    'rex': (
        'rajpurkarlab/ReXGroundingCT',
        '662d836108a807d47a40421ac5ed4d1c7e133b16',
        ('dataset.json', 'MICCAI_challenge_dataset.json'),
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--download', action='store_true',
                        help='Use the machine’s existing Hugging Face access and fetch the four metadata files.')
    args = parser.parse_args()
    inventory = []
    for label, (repo, revision, filenames) in FILES.items():
        for filename in filenames:
            record = dict(repo=repo, revision=revision, file=filename)
            if args.download:
                try:
                    path = hf_hub_download(repo_id=repo, repo_type='dataset', revision=revision,
                                           filename=filename, local_dir=args.destination / label)
                except GatedRepoError:
                    parser.exit(2, f'Access is required for {repo}. Check the logged-in account on this machine.\n')
                record['local_path'] = str(Path(path).resolve())
                record['bytes'] = Path(path).stat().st_size
            inventory.append(record)
    output = json.dumps({'downloaded': args.download, 'files': inventory}, indent=2)
    if args.download:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.destination, delete=False) as f:
            f.write(output + '\n')
        os.replace(f.name, args.destination / 'source_versions.json')
    print(output)


if __name__ == '__main__':
    main()
