
def visualize_cmap_value(cmap, value, vmin=0.0, vmax=1.0, label=True):
    """
    cmap: str or Colormap (e.g. 'viridis')
    value: scalar to visualize
    vmin, vmax: normalization range
    """
    cmap_obj = mpl.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap_obj(norm(value))

    fig, ax = plt.subplots(figsize=(4, 1))
    ax.imshow(np.array([[rgba]]), aspect='auto')
    ax.set_axis_off()

    if label:
        ax.set_title(f"{cmap_obj.name}: value={value}, rgba={tuple(round(x, 3) for x in rgba)}", fontsize=10)

    plt.show()
    return rgba