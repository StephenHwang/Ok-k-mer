"""Run inside the image: python /tests/docker_smoke.py /data/smoke-output.

Exercises the real bundled Dashing2 and CLI, with no external input downloads.
The output directory must not already exist; it is retained for inspection.
"""
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(42)
    ancestor = ''.join(rng.choices('ACGT', k=2000))
    for name, rate in zip('abcde', [0, .01, .03, .08, .15]):
        sequence = ''.join(rng.choice('ACGT'.replace(base, ''))
                           if rng.random() < rate else base for base in ancestor)
        (root / f'{name}.fna').write_text(f'>{name}\n{sequence}\n')
    (root / 'genomes.tsv').write_text('genome_id\tpath\n' + ''.join(
        f'{name}\t{name}.fna\n' for name in 'abcde'))
    (root / 'reference.nwk').write_text('((a:1,b:2):1,(c:3,d:4):2,e:6);\n')

    def run(*args):
        subprocess.run(['okkmer', *map(str, args)], check=True)

    run('--version')
    run('index', '--genomes', root / 'genomes.tsv', '--k-low', 6, '--k-high', 8,
        '--threads', 2, '--output', root / 'index')
    run('search', '--tree', root / 'reference.nwk', '--index', root / 'index',
        '--k-low', 6, '--k-high', 8, '-n', 2, '--w-step', .5,
        '--annotate-nodes', '--no-early-stop', '--output', root / 'results')
    run('plot', 'heatmap', '--results', root / 'results', '--ensemble-size', 2,
        '--output', root / 'heatmap.pdf')
    run('plot', 'tree', '--results', root / 'results', '--min-leaves', 2,
        '--output', root / 'tree.svg')
    result = json.loads((root / 'results/run.json').read_text())
    assert result['n_pairs'] == 10, result
    for name in ['heatmap.pdf', 'tree.svg', 'results/recommendations.tsv',
                 'index/dashing2/jaccard_similarity_k6.tsv']:
        assert (root / name).stat().st_size > 0, name
    print(f'Container smoke test passed. Outputs: {root}')


if __name__ == '__main__':
    main()
