"""Render search outputs without repeating optimization or requiring Qt."""
import json
from pathlib import Path
import numpy as np
from .io import read_table


def pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError('Plotting requires the optional dependency: pip install "okkmer[plot]"') from exc
    return plt


def save(fig, output, plt):
    path = Path(output)
    if path.exists():
        raise ValueError(f'Refusing to overwrite existing figure: {path}')
    if path.suffix.lower() not in ('.pdf','.png','.svg'):
        raise ValueError('Plot output must end in .pdf, .png, or .svg')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def heatmap(args):
    plt = pyplot()
    root = Path(args.results)
    metadata = json.loads((root/'run.json').read_text())
    if args.ensemble_size == 3 and args.fixed_k is None:
        raise ValueError('A three-length heatmap requires --fixed-k for a two-dimensional slice')
    if args.ensemble_size != 3 and args.fixed_k is not None:
        raise ValueError('--fixed-k is only valid for three-length heatmaps')
    rows = read_table(root/f'combinations_{args.ensemble_size}.tsv')
    ks = metadata['ks']
    if args.fixed_k is not None and args.fixed_k not in ks:
        raise ValueError('--fixed-k is outside the search k range')
    axis_ks = [k for k in ks if k != args.fixed_k]
    lookup = {k:i for i,k in enumerate(axis_ks)}
    matrix = np.full((1,len(ks)) if args.ensemble_size == 1 else (len(axis_ks),len(axis_ks)),np.nan)
    for row in rows:
        lengths = json.loads(row['ks'])
        if args.ensemble_size == 1:
            matrix[0,lookup[lengths[0]]] = float(row['rho'])
        else:
            if args.ensemble_size == 3:
                if args.fixed_k not in lengths:
                    continue
                lengths.remove(args.fixed_k)
            a,b = (lookup[k] for k in lengths)
            matrix[a,b] = matrix[b,a] = float(row['rho'])
    if not np.isfinite(matrix).any():
        raise ValueError('No evaluated combinations are available in this heatmap slice')
    fig,ax = plt.subplots(figsize=(8,2.5 if args.ensemble_size == 1 else 7))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    im = ax.imshow(matrix,origin='lower',cmap=cmap,aspect='auto' if args.ensemble_size == 1 else 'equal',vmin=-1,vmax=1)
    ax.set_xticks(range(len(axis_ks)),axis_ks,rotation=90)
    if args.ensemble_size == 1:
        ax.set_yticks([])
        ax.set_xlabel('k-mer length')
    else:
        ax.set_yticks(range(len(axis_ks)),axis_ks)
        ax.set(xlabel='k-mer length',ylabel='k-mer length')
    title = f'{args.ensemble_size}-length ensembles: best rho over weights'
    if args.fixed_k is not None:
        title += f' (fixed k={args.fixed_k})'
    ax.set_title(title+'\nGray: unavailable, unsearched, or repeated lengths',fontsize=10)
    fig.colorbar(im,ax=ax,label='Spearman correlation, ρ')
    fig.tight_layout()
    save(fig,args.output,plt)


def tree(args):
    from .search import load_reference
    plt = pyplot()
    root = Path(args.results)
    path = root/'nodes.json'
    if not path.exists():
        raise ValueError('Tree annotations require search results generated with --annotate-nodes')
    metadata = json.loads((root/'run.json').read_text())
    annotations = {row['node_id']:row for row in json.loads(path.read_text())}
    reference = load_reference(root/'reference.nwk')
    excluded = set(metadata['excluded_genomes'])
    leaf_order = [leaf for leaf in reference if leaf.name not in excluded]
    ypos = {leaf:float(i) for i,leaf in enumerate(leaf_order)}
    xpos = {reference:0.}
    for node in reference.traverse('preorder'):
        if node is not reference:
            xpos[node] = xpos[node.up]+node.dist
    for node in reference.traverse('postorder'):
        if not node.is_leaf():
            ys = [ypos[child] for child in node.children if child in ypos]
            if ys:
                ypos[node] = (min(ys)+max(ys))/2
    fig,ax = plt.subplots(figsize=(12,max(4,len(leaf_order)*.18)))
    for node in reference.traverse():
        if node not in ypos:
            continue
        children = [c for c in node.children if c in ypos]
        if children:
            ax.plot([xpos[node]]*2,[min(ypos[c] for c in children),max(ypos[c] for c in children)],color='gray',lw=.7)
        for child in children:
            ax.plot([xpos[node],xpos[child]],[ypos[child]]*2,color='gray',lw=.7)
    for leaf in leaf_order:
        ax.annotate(leaf.name,(xpos[leaf],ypos[leaf]),xytext=(4,0),textcoords='offset points',va='center',fontsize=7)
    for i,node in enumerate(n for n in reference.traverse('preorder') if not n.is_leaf()):
        row = annotations.get(f'node_{i:06d}',{})
        best = row.get('recommendation')
        if best and node in ypos:
            text = ','.join(map(str,best['ks'])) + f"; ρ={best['rho']:.3f}"
            ax.annotate(text,(xpos[node],ypos[node]),xytext=(3,5),textcoords='offset points',fontsize=6,color='navy')
    ax.set(xlabel='Reference branch distance',yticks=[])
    ax.spines[['top','right','left']].set_visible(False)
    ax.margins(x=.25,y=.02)
    fig.tight_layout()
    save(fig,args.output,plt)
