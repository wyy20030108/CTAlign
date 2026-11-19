"""Verify archived CT experiment sources and optionally reconstruct an archive's source layout."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(studies, archive_directory):
    verified = set()
    for study, record in studies.items():
        entries = record['sources']
        for entry, item in entries.items():
            source = REPO / item['path']
            if source not in verified:
                if sha256(source) != item['sha256']:
                    raise ValueError(f'Source SHA-256 mismatch: {source}')
                if source.suffix == '.py':
                    compile(source.read_bytes(), str(source), 'exec')
                verified.add(source)
        if archive_directory:
            archive_path = archive_directory / record['archive']
            if sha256(archive_path) != record['archive_sha256']:
                raise ValueError(f'Archive SHA-256 mismatch: {archive_path}')
            with ZipFile(archive_path) as archive:
                original = {entry for entry in archive.namelist() if Path(entry).suffix in ('.py', '.sh')}
                if set(entries) != original:
                    raise ValueError(f'Historic source inventory mismatch: {study}')
                for entry, item in entries.items():
                    expected = item.get('archive_sha256', item['sha256'])
                    if hashlib.sha256(archive.read(entry)).hexdigest() != expected:
                        raise ValueError(f'Archive member mismatch: {study}/{entry}')
    return len(verified)


def materialize(study, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for entry, data in study['sources'].items():
        target = destination / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / data['path']).read_bytes())


def main():
    catalog = json.loads((HERE / 'source-index.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=catalog)
    parser.add_argument('--archives', type=Path, help='Optional directory containing the original local ZIPs')
    parser.add_argument('--destination', type=Path, help='New directory for the selected study source layout')
    options = parser.parse_args()
    if options.destination and not options.study:
        parser.error('--destination requires --study')
    choices = {options.study: catalog[options.study]} if options.study else catalog
    count = verify(choices, options.archives)
    print(f'Verified {count} unique source files in {len(choices)} studies.')
    if options.destination:
        materialize(catalog[options.study], options.destination)


if __name__ == '__main__':
    main()
