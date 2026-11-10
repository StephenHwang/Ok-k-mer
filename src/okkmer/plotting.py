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


def _heatmap_data(rows, ks, size, fixed_k=None, lower_rows=()):
    """Keep winning rho and matching weights together, including mirrored cells."""
    lookup = {k:i for i,k in enumerate(ks)}
    shape = (1,len(ks)) if size == 1 else (len(ks),len(ks))
    matrix = np.full(shape,np.nan)
    weights = np.full(shape,np.nan)
    for row in rows:
        lengths = json.loads(row['ks'])
        rho = float(row['rho'])
        if not np.isfinite(rho):
            continue
        if size == 1:
            cells = [(0,lookup[lengths[0]],np.nan)]
        else:
            if size == 3 and fixed_k not in lengths:
                continue
            saved_weights = json.loads(row['weights'])
            by_k = dict(zip(lengths,saved_weights))
            remaining = sorted(k for k in lengths if k != fixed_k)
            a,b = (lookup[k] for k in remaining)
            # k1 is the row in 2k plots, or the fixed length in 3k plots.
            cells = [(a,b,by_k[fixed_k] if size == 3 else by_k[remaining[0]]),
                     (b,a,by_k[fixed_k] if size == 3 else by_k[remaining[1]])]
        for y,x,weight in cells:
            if not np.isfinite(matrix[y,x]) or rho > matrix[y,x]:
                matrix[y,x],weights[y,x] = rho,weight
    # Repeated lengths reduce to a saved lower-order ensemble. Do not use
    # lower-order scores to fill unsearched combinations of distinct lengths.
    reduced = {}
    for row in lower_rows:
        key = tuple(sorted(json.loads(row['ks'])))
        rho = float(row['rho'])
        if len(key) < size and np.isfinite(rho):
            reduced[key] = max(reduced.get(key,-np.inf),rho)
    if size > 1:
        for y,k_y in enumerate(ks):
            for x,k_x in enumerate(ks):
                key = tuple(sorted({k_y,k_x} if size == 2 else {fixed_k,k_y,k_x}))
                if len(key) < size and key in reduced:
                    matrix[y,x] = reduced[key]
                    # A lower-order result has no unique split of its weights
                    # between duplicate positions; show correlation only.
                    weights[y,x] = np.nan
    return matrix,weights


def _heatmap_figure(rows, metadata, size, fixed_ks, show_weights, plt, vmax=None, lower_rows=()):
    """Notebook-style correlation panels with optional triangular weight circles."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Circle

    ks = metadata['ks']
    panels = [_heatmap_data(rows,ks,size,k,lower_rows) for k in fixed_ks]
    finite = [matrix[np.isfinite(matrix)] for matrix,_ in panels]
    finite = np.concatenate(finite)
    if not finite.size:
        raise ValueError('No evaluated combinations are available in this heatmap slice')
    if vmax is None:
        vmax = max(float(finite.max()),1e-12)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('lightgray')
    norm = Normalize(0,vmax)
    # The original one-k plot deliberately fills the same square as the 2k plot.
    n = len(panels)
    width = (8 if size != 3 else max(6,3.5*n+1.2)) + (1 if show_weights else 0)
    height = 6 if size != 3 else 4.5
    fig = plt.figure(figsize=(width,height),layout='constrained')
    grid = fig.add_gridspec(1,n+1+int(show_weights),
                            width_ratios=[1]*n+[.045]+([.045] if show_weights else []))
    weight_cmap = plt.get_cmap('Greys',max(2,round(1/metadata.get('w_step',.05))+1))
    weight_norm = Normalize(0,1)
    for i,((matrix,weights),fixed_k) in enumerate(zip(panels,fixed_ks)):
        ax = fig.add_subplot(grid[0,i])
        im = ax.imshow(matrix,origin='lower',cmap=cmap,norm=norm,
                       aspect='auto' if size == 1 else 'equal')
        ax.set_box_aspect(1)
        ticks = list(range(0,len(ks),2 if size == 3 else 1))
        labels = [str(ks[j]) if i % 2 == 0 else '' for i,j in enumerate(ticks)]
        ax.set_xticks(ticks,labels,fontsize=13)
        ax.set_xlabel('$k_1$' if size == 1 else ('$k_2$' if size == 2 else '$k_3$'),fontsize=17)
        if size == 1:
            ax.set_yticks([])
        else:
            ax.set_yticks(ticks,labels,fontsize=13)
            ax.set_ylabel('$k_1$' if size == 2 else '$k_2$',fontsize=17)
        if size == 3:
            ax.set_title(f'$k_1$={fixed_k}',fontsize=17)
        if show_weights:
            for y,x in zip(*np.where(np.isfinite(matrix) & np.isfinite(weights))):
                if x > y:
                    ax.add_patch(Circle((x,y),.4,facecolor=weight_cmap(weight_norm(weights[y,x])),
                                        edgecolor='black',linewidth=.5))
    cbar = fig.colorbar(im,cax=fig.add_subplot(grid[0,n]))
    cbar.set_label('Spearman correlation, ρ',fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    if show_weights:
        bar = fig.colorbar(ScalarMappable(norm=weight_norm,cmap=weight_cmap),
                           cax=fig.add_subplot(grid[0,n+1]),ticks=np.linspace(0,1,6))
        bar.set_label('$k_1$ weight',fontsize=16)
        bar.ax.tick_params(labelsize=13)
    words = {1:'one k-mer',2:'two k-mers',3:'three k-mers'}
    fig.suptitle(f"Spearman correlation for {words[size]}",fontsize=19)
    return fig


def heatmap(args):
    plt = pyplot()
    root = Path(args.results)
    metadata = json.loads((root/'run.json').read_text())
    if args.ensemble_size == 3 and args.fixed_k is None:
        raise ValueError('A three-length heatmap requires --fixed-k for a two-dimensional slice')
    if args.ensemble_size != 3 and args.fixed_k is not None:
        raise ValueError('--fixed-k is only valid for three-length heatmaps')
    if args.k_radius < 0 or (args.k_radius and args.ensemble_size != 3):
        raise ValueError('--k-radius must be nonnegative and is only valid for three-length heatmaps')
    if args.show_weights and args.ensemble_size == 1:
        raise ValueError('--show-weights is only valid for two- or three-length heatmaps')
    if args.vmax is not None and (not np.isfinite(args.vmax) or args.vmax <= 0):
        raise ValueError('--vmax must be finite and positive')
    if args.fixed_k is not None and args.fixed_k not in metadata['ks']:
        raise ValueError('--fixed-k is outside the search k range')
    rows = read_table(root/f'combinations_{args.ensemble_size}.tsv')
    lower_rows = []
    for size in range(1,args.ensemble_size):
        path = root/f'combinations_{size}.tsv'
        if size == 1 and not path.exists():
            path = root/'single_k.tsv'
        if path.exists():
            lower_rows.extend(read_table(path))
    fixed_ks = ([k for k in metadata['ks'] if abs(k-args.fixed_k) <= args.k_radius]
                if args.ensemble_size == 3 else [None])
    fig = _heatmap_figure(rows,metadata,args.ensemble_size,fixed_ks,args.show_weights,plt,args.vmax,lower_rows=lower_rows)
    try:
        save(fig,args.output,plt)
    finally:
        plt.close(fig)


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
