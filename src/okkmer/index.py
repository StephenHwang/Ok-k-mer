"""Prepare complete all-pairs distance datasets, independently of a tree."""
import re
import shutil
import subprocess
from pathlib import Path
from itertools import chain

import numpy as np

from . import __version__
from .io import digest, iter_table, new_directory, read_table, utc_now, write_json


def validate_bounds(low, high):
    if low < 1 or high < low:
        raise ValueError('Require 1 <= k-low <= k-high (inclusive)')


def read_genomes(path):
    rows = read_table(path)
    genomes = {}
    for row in rows:
        name, source = row.get('genome_id', ''), row.get('path', '')
        if not name or not source or name in genomes:
            raise ValueError('Genome manifest requires unique genome_id and path columns')
        source = Path(source)
        if not source.is_absolute():
            source = Path(path).resolve().parent / source
        source = source.resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f'Missing or empty genome: {source}')
        if '\n' in str(source) or '\t' in str(source):
            raise ValueError('Genome paths cannot contain tabs or newlines')
        genomes[name] = source
    if len(genomes) < 3 or len(set(genomes.values())) != len(genomes):
        raise ValueError('Require at least three genomes, each with a distinct input path')
    return genomes


def wide_pairs(path, ks):
    rows = iter_table(path)
    first = next(rows, None)
    if first is None:
        raise ValueError('Distance table is empty')
    columns = []
    for k in ks:
        candidates = [c for c in (f'k{k}', f'k{k}_distance') if c in first]
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one column k{k} or k{k}_distance')
        columns.append(candidates[0])
    names = set()
    pairs = {}
    for row in chain([first], rows):
        a, b = row.get('genome_1', ''), row.get('genome_2', '')
        if not a or not b or a == b:
            raise ValueError('Require genome_1 and genome_2 with distinct, nonempty IDs')
        key = tuple(sorted((a, b)))
        if key in pairs:
            raise ValueError(f'Duplicate unordered pair: {key}')
        pairs[key] = [float(row[c]) for c in columns]
        names.update(key)
    names = sorted(names)
    if len(names) < 3 or len(pairs) != len(names) * (len(names)-1) // 2:
        raise ValueError('Input must contain every unordered genome pair exactly once')
    ordered = [(a, b) for i, a in enumerate(names) for b in names[i+1:]]
    return names, np.asarray([pairs[p] for p in ordered], dtype=float)


def square_matrix(path, label_map=None):
    label_map = label_map or {}
    columns, rows = None, {}
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('#Sources\t'):
                if columns is not None:
                    raise ValueError(f'Duplicate #Sources header: {path}')
                columns = [label_map.get(v, v) for v in line.rstrip('\r\n').split('\t')[1:]]
            elif line.strip() and not line.startswith('#'):
                fields = line.rstrip('\r\n').split('\t')
                name = label_map.get(fields[0], fields[0])
                if name in rows:
                    raise ValueError(f'Duplicate matrix row: {name}')
                rows[name] = [float(v) for v in fields[1:]]
    if columns is None or len(columns) != len(set(columns)) or set(rows) != set(columns):
        raise ValueError(f'Expected a labeled square matrix with a #Sources header: {path}')
    if any(len(row) != len(columns) for row in rows.values()):
        raise ValueError(f'Incomplete square matrix: {path}')
    names = sorted(columns)
    lookup = {name: i for i, name in enumerate(columns)}
    matrix = np.asarray([[rows[a][lookup[b]] for b in names] for a in names])
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-8, rtol=0):
        raise ValueError(f'Nonfinite or asymmetric matrix: {path}')
    return names, matrix


def matrix_directory(path, ks, kind, label_map=None):
    names, values, sources = None, [], []
    for k in ks:
        stem = 'jaccard_similarity' if kind == 'similarity' else 'jaccard'
        source = Path(path) / f'{stem}_k{k}.tsv'
        current, matrix = square_matrix(source, label_map)
        if not np.allclose(matrix.diagonal(), 1 if kind == 'similarity' else 0, atol=1e-8, rtol=0):
            raise ValueError(f'Unexpected matrix diagonal: {source}')
        if names is not None and current != names:
            raise ValueError('Genome IDs differ between k matrices')
        names = current
        values.append(matrix[np.triu_indices(len(names), 1)])
        sources.append({'path': str(source.resolve()), 'sha256': digest(source)})
    return names, np.asarray(values).T, sources


def build(args):
    validate_bounds(args.k_low, args.k_high)
    ks = list(range(args.k_low, args.k_high+1))
    if args.threads < 1:
        raise ValueError('threads must be positive')
    provenance = {'source_kind': 'genomes' if args.genomes else 'distances'}
    label_map = {}
    if args.genome_map:
        for row in read_table(args.genome_map):
            source, target = row.get('source_id'), row.get('genome_id')
            if not source or not target or source in label_map:
                raise ValueError('Genome map requires unique source_id and nonempty genome_id')
            label_map[source] = target
        if len(set(label_map.values())) != len(label_map):
            raise ValueError('Genome map must be one-to-one')
        provenance['genome_map'] = {'path': str(Path(args.genome_map).resolve()), 'sha256': digest(args.genome_map)}
    if args.genomes:
        if args.genome_map or args.input_kind != 'distance':
            raise ValueError('--genome-map and --input-kind apply only to imported distances')
        genomes = read_genomes(args.genomes)
        executable = shutil.which(args.dashing2)
        if not executable:
            raise ValueError(f'Dashing2 executable not found: {args.dashing2}')
        version = subprocess.run([executable, '--version'], capture_output=True, text=True)
        description = (version.stdout + version.stderr).strip()
        # v2.1.21 prints its version banner but exits 1 for --version.
        match = re.search(r'Dashing2\s+(?:version\s+)?v?([\w.]+)', description, re.IGNORECASE)
        revision = match.group(1) if match else None
        accepted = {'2.1.21', '72d16c72fdb43f37cff06f8a13c94b7fdc9d29af'}
        if version.returncode not in (0, 1) or revision not in accepted:
            raise ValueError(f'Expected Dashing2 v2.1.21 or the pinned manuscript revision; --version returned: {description}')
        out = new_directory(args.output)
        raw = out / 'dashing2'
        raw.mkdir()
        # Dashing's --outprefix flattens input basenames into one cache folder.
        # Use ID-based aliases; the ordinal also separates sanitized/truncated IDs.
        inputs = (raw / 'genomes').resolve()
        inputs.mkdir()
        staged_genomes = {}
        for i, (name, source) in enumerate(genomes.items()):
            safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', name)[:100]
            suffix = source.suffix
            if not re.fullmatch(r'\.[A-Za-z][A-Za-z0-9]{0,15}', suffix):
                suffix = '.fna'
            staged = inputs / f'{i:06d}_{safe_id}{suffix}'
            staged.symlink_to(source)
            staged_genomes[name] = staged
        genome_list = raw / 'genome_paths.txt'
        genome_list.write_text(''.join(str(path)+'\n' for path in staged_genomes.values()))
        commands = []
        for k in ks:
            work = raw / f'work_k{k}'
            work.mkdir()
            cmd = [executable, 'dist', '--set', '--long-kmers', '-k', str(k), '-p', str(args.threads),
                   '--square', '--outprefix', str(work.resolve()), '-F', str(genome_list.resolve())]
            commands.append(cmd)
            with (raw/f'jaccard_similarity_k{k}.tsv').open('w') as stdout, (raw/f'k{k}.stderr').open('w') as stderr:
                subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True, cwd=work.resolve())
            # Remove only successful invocation's private intermediates.
            shutil.rmtree(work)
        names, values, sources = matrix_directory(raw, ks, 'similarity', {str(p): n for n,p in staged_genomes.items()})
        kind = 'similarity'
        provenance.update(dashing2_version=description, dashing2_revision=revision, executable=str(Path(executable).resolve()),
                          executable_sha256=digest(executable), commands=commands,
                          genomes=[{'genome_id': n, 'path': str(p), 'sha256': digest(p),
                                    'dashing2_path': str(staged_genomes[n])} for n,p in genomes.items()])
    else:
        source = Path(args.distances)
        kind = args.input_kind
        if source.is_dir():
            names, values, sources = matrix_directory(source, ks, kind, label_map)
        else:
            names, values = wide_pairs(source, ks)
            if label_map:
                raise ValueError('--genome-map currently applies to Dashing2 square matrix imports')
            sources = [{'path': str(source.resolve()), 'sha256': digest(source)}]
        out = new_directory(args.output)
    if len(names) < 3 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Require >=3 genomes and finite Jaccard values in [0,1]')
    if kind == 'similarity':
        values = 1-values
    np.savez_compressed(out/'distances.npz', distances=values)
    manifest = {'schema_version': 1, 'okkmer_version': __version__, 'created_utc': utc_now(),
                'metric': 'jaccard_distance', 'genomes': names, 'ks': ks,
                'pair_order': 'upper_triangle_row_major', 'n_pairs': len(values),
                'distances_sha256': digest(out/'distances.npz'), 'input_kind': kind,
                'sources': sources, 'provenance': provenance}
    write_json(out/'manifest.json', manifest)
    print(f'Prepared {len(names)} genomes, {len(values)} pairs, k={ks[0]}–{ks[-1]}: {out}')


def load(path):
    import json
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('metric') != 'jaccard_distance':
        raise ValueError('Unsupported index schema or metric')
    if manifest.get('pair_order') != 'upper_triangle_row_major':
        raise ValueError('Unsupported index pair order')
    if digest(path/'distances.npz') != manifest['distances_sha256']:
        raise ValueError('Index distance checksum mismatch')
    with np.load(path/'distances.npz', allow_pickle=False) as archive:
        values = archive['distances']
    names, ks = manifest['genomes'], manifest['ks']
    if len(names) != len(set(names)) or len(ks) != len(set(ks)):
        raise ValueError('Duplicate genome IDs or k values in index')
    if values.shape != (len(names)*(len(names)-1)//2,len(ks)) or not np.isfinite(values).all() or np.any((values < 0)|(values > 1)):
        raise ValueError('Invalid index distance array')
    return manifest, values
