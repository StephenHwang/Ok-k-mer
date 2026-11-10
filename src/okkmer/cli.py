"""Command-line interface; heavy dependencies load only for the chosen command."""
import argparse
import sys
from pathlib import Path
import subprocess

from . import __version__


def parser():
    p = argparse.ArgumentParser(prog='okkmer',description='Ok-k-mer selects weighted ensembles of k-mer lengths that maximize Spearman correlation with a reference phylogeny.')
    p.add_argument('--version',action='version',version=f'okkmer {__version__}')
    sub = p.add_subparsers(dest='command',metavar='{index,search,plot,help}')
    index = sub.add_parser('index',help='Compute or import all-pairs Jaccard distances')
    source = index.add_mutually_exclusive_group(required=True)
    source.add_argument('--genomes',type=Path,help='TSV/CSV with genome_id and path columns')
    source.add_argument('--distances',type=Path,help='Wide pair table or directory of Dashing2 square TSV matrices')
    index.add_argument('--input-kind',choices=['distance','similarity'],default='distance',help='Imported value semantics (default: distance)')
    index.add_argument('--genome-map',type=Path,help='Explicit source_id/genome_id TSV mapping for square matrices')
    index.add_argument('--dashing2',default='dashing2',help='Dashing2 v2.1.21 executable (default: dashing2 on PATH)')
    bounds(index)
    index.add_argument('--threads',type=int,default=1)
    index.add_argument('--output',type=Path,required=True)
    search = sub.add_parser('search',help='Select whole-dataset and optionally subtree ensembles')
    search.add_argument('--tree',type=Path,required=True)
    search.add_argument('--index',type=Path,required=True)
    mode = search.add_mutually_exclusive_group()
    mode.add_argument('--global',dest='global_search',action='store_true',help='Search all distinct length combinations (default)')
    mode.add_argument('--local',type=int,nargs='?',const=3,metavar='DELTA_K',help='Vary preceding lengths within ±DELTA_K (default: 3); additional length spans full range')
    search.add_argument('--max-kmers','-n',type=int,default=3,help='Maximum ensemble size (default: 3); stops early without improvement')
    bounds(search)
    search.add_argument('--w-step',type=float,default=.05,help='Weight step dividing 1 exactly (default: 0.05)')
    search.add_argument('--threads',type=int,default=1)
    stopping = search.add_mutually_exclusive_group()
    stopping.add_argument('--no-early-stop',action='store_true',help='Evaluate every ensemble size through --max-kmers and select the highest rho')
    stopping.add_argument('--improvement-tol',type=float,default=1e-12,help='Required rho improvement for early stopping (default: 1e-12)')
    search.add_argument('--exclude-genomes',type=Path,help='One exact genome ID per line to exclude from comparisons and tree annotations')
    search.add_argument('--save-grid',action='store_true',help='Also save every evaluated whole-dataset weight score (potentially large)')
    search.add_argument('--annotate-nodes',action='store_true',help='Repeat selection at internal nodes; can be expensive')
    search.add_argument('--dry-run',action='store_true',help='Validate inputs and print candidate counts without searching or writing outputs')
    search.add_argument('--output',type=Path,required=True)
    plot = sub.add_parser('plot',help='Plot saved results without repeating searches')
    plots = plot.add_subparsers(dest='plot_command',required=True)
    heat = plots.add_parser('heatmap',help='Best rho over weights; triples require a fixed-k slice')
    heat.add_argument('--ensemble-size',type=int,choices=[1,2,3],default=2)
    heat.add_argument('--fixed-k',type=int)
    tree = plots.add_parser('tree',help='Render saved subtree annotations')
    for item in (heat,tree):
        item.add_argument('--results',type=Path,required=True)
        item.add_argument('--output',type=Path,required=True,help='New .pdf, .png, or .svg file')
    sub.add_parser('help',help='Show help; also accepts a subcommand, e.g. okkmer help search')
    return p


def bounds(p):
    p.add_argument('--k-low',type=int,default=6,help='Inclusive minimum k (default: 6)')
    p.add_argument('--k-high',type=int,default=40,help='Inclusive maximum k (default: 40)')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    if not argv:
        p.print_help()
        return 0
    if argv[0] == 'help':
        argv = argv[1:]+['--help']
    args = p.parse_args(argv)
    try:
        if args.command == 'index':
            from .index import build
            build(args)
        elif args.command == 'search':
            from .search import run
            run(args)
        elif args.command == 'plot':
            from . import plotting
            getattr(plotting,args.plot_command)(args)
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        print(f'okkmer: error: {exc}',file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('okkmer: interrupted',file=sys.stderr)
        return 130
    return 0
