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
        raise ValueError('Matplotlib is missing; reinstall Ok-k-mer with its required dependencies.') from exc
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


def _tree_figure(reference, annotations, metadata, plt, min_leaves=5):
    """Adapt the notebook's weighted k pies and labels without Qt tree faces.

    CLI rho already has the distance-correlation sign (unlike the notebook's
    negative stored correlation), so it must be displayed without negation.
    Node IDs refer to the original preorder, before excluded leaves are hidden.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Wedge

    excluded = set(metadata.get('excluded_genomes', []))
    leaves = [leaf for leaf in reference if leaf.name not in excluded]
    if not leaves:
        raise ValueError('No retained genomes to plot')
    ypos = {leaf:float(i)*.8 for i,leaf in enumerate(reversed(leaves))}
    counts = {leaf:1 for leaf in leaves}
    for node in reference.traverse('postorder'):
        children = [child for child in node.children if child in ypos]
        if children:
            ypos[node] = (ypos[children[0]]+ypos[children[-1]])/2
            counts[node] = sum(counts[child] for child in children)
    # The threshold controls recommendation faces only; keep the full retained
    # topology in ypos/counts so smaller clades and their leaves are still drawn.
    best_by_node = {}
    for i,node in enumerate(n for n in reference.traverse('preorder') if not n.is_leaf()):
        best = annotations.get(f'node_{i:06d}',{}).get('recommendation')
        if best and counts.get(node,0) >= min_leaves:
            best_by_node[node] = best

    ks = sorted(metadata['ks'])
    low,high = min(ks),max(ks)
    # Use every integer in the search range, as in make_k_color in the notebook.
    colors = [plt.get_cmap('viridis')((k-low)/max(high-low,1))
              for k in range(low,high+1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(low-.5,high+1.5),cmap.N)
    depths = reference.get_farthest_leaf()[1]
    branch_scale = 6/max(depths,1e-12)
    annotation_padding = 3.5
    xpos = {reference:0.}
    for node in reference.traverse('preorder'):
        if node is not reference:
            # Reserve space for each node face, as ete3 does. This preserves
            # topology and branch-length contributions but adds face padding.
            padding = annotation_padding if node.up in best_by_node else .12
            xpos[node] = xpos[node.up]+node.dist*branch_scale+padding
    width = max(xpos[n]+(annotation_padding if n in best_by_node else 0) for n in ypos)+2.5
    height = max(3.5,len(leaves)*.8+1.5)
    fig,ax = plt.subplots(figsize=(width,height))
    for node in reference.traverse():
        if node not in ypos:
            continue
        children = [child for child in node.children if child in ypos]
        if children:
            ax.plot([xpos[node]]*2,[min(ypos[c] for c in children),max(ypos[c] for c in children)],color='black',lw=1)
        for child in children:
            ax.plot([xpos[node],xpos[child]],[ypos[child]]*2,color='black',lw=1)
    for leaf in leaves:
        ax.annotate(leaf.name,(xpos[leaf],ypos[leaf]),xytext=(4,0),textcoords='offset points',va='center',fontsize=13)
    for node,best in best_by_node.items():
        weights = np.asarray(best['weights'],dtype=float)
        if len(weights) != len(best['ks']) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Invalid recommendation weights for tree annotation')
        # 38 points is approximately the original ETE face's 50 pixels at 96 dpi.
        pie = DrawingArea(38,38,0,0)
        angle = 0.
        for k,weight in zip(best['ks'],weights/weights.sum()):
            end = angle+360*weight
            pie.add_artist(Wedge((19,19),19,angle,end,facecolor=cmap(norm(k)),edgecolor='none'))
            angle = end
        ax.add_artist(AnnotationBbox(pie,(xpos[node],ypos[node]),xybox=(22,0),
                                    xycoords='data',boxcoords='offset points',frameon=False,pad=0))
        label = ('k='+','.join(map(str,best['ks']))+
                 '\nw='+','.join(f'{w:.3g}' for w in weights)+
                 f"\nρ={best['rho']:.3f}")
        ax.annotate(label,(xpos[node],ypos[node]),xytext=(46,0),
                    textcoords='offset points',va='center',fontsize=15)
    ax.set_xlim(-.15,width)
    ax.set_ylim(-.6,max(ypos.values())+.6)
    ax.set_axis_off()
    # A separate centered axis keeps the legend clear of leaf/node labels.
    fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.9/height)
    legend_ax = fig.add_axes([.25,.22/height,.5,.16/height])
    ticks = list(range(low,high+1,2))
    if high not in ticks:
        ticks.append(high)
    bar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=legend_ax,
                      orientation='horizontal',ticks=ticks)
    bar.set_label('k-mer length')
    return fig


def tree(args):
    from .search import load_reference
    plt = pyplot()
    root = Path(args.results)
    path = root/'nodes.json'
    if not path.exists():
        raise ValueError('Tree annotations require search results generated with --annotate-nodes')
    if args.dpi <= 0 or args.min_leaves <= 0:
        raise ValueError('--dpi and --min-leaves must be positive')
    output = Path(args.output)
    outputs = [output] if output.suffix else [output.with_suffix('.pdf'),output.with_suffix('.png')]
    # Preflight both outputs before writing either file.
    for target in outputs:
        if target.suffix.lower() not in ('.pdf','.png','.svg'):
            raise ValueError('Plot output must end in .pdf, .png, or .svg, or be a prefix without an extension')
        if target.exists():
            raise ValueError(f'Refusing to overwrite existing figure: {target}')
    metadata = json.loads((root/'run.json').read_text())
    annotations = {row['node_id']:row for row in json.loads(path.read_text())}
    reference = load_reference(root/'reference.nwk')
    fig = _tree_figure(reference,annotations,metadata,plt,args.min_leaves)
    try:
        for target in outputs:
            target.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(target,dpi=args.dpi,bbox_inches='tight')
            print(f'Saved {target}')
    finally:
        plt.close(fig)
