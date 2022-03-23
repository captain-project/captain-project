import os, sys, glob
import pickle
import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=1, precision=3)  # prints floats, no scientific notation
from matplotlib.patches import Rectangle, Polygon
from matplotlib.gridspec import GridSpec
import baltic as bt
from ..biodivinit.PhyloGenerator import convert_to_bt_tree


def plot_species_ranges_list(
    pklfile=None,
    loaded_env=None,
    species_list=[6, 12, 200, 350],
    log_transform=1,
    plot_titles=True,
):
    if loaded_env:
        env = loaded_env
    else:
        with open(pklfile, "rb") as pkl:
            env = pickle.load(pkl)
    evolveGrid = env.bioDivGrid
    resolution = env.resolution

    pop_sp = evolveGrid.individualsPerSpecies()
    range_sp = evolveGrid.geoRangePerSpecies()
    max_pop_sp = []
    # main_ttl = "Time: %s" % evolveGrid._counter
    abundance_map = []
    ttl = []
    titles = []
    for sp_ID in species_list:
        ttl.append(
            "Sp. %s (pop. size: %s, range size: %s)"
            % (sp_ID, round(pop_sp[sp_ID]), round(range_sp[sp_ID]))
        )
        titles.append(f"Sp. {sp_ID}")
        if log_transform:
            abundance_map.append(np.log10(1 + evolveGrid._h[sp_ID]))
            max_pop_sp.append(np.max(np.log10(1 + evolveGrid._h[sp_ID])))
        else:
            abundance_map.append(evolveGrid._h[sp_ID])
            max_pop_sp.append(np.max(evolveGrid._h[sp_ID]))

    fontsize = 15

    q_indx = env.protected_quadrants
    x_coord, y_coord = [], []
    for i in q_indx:
        x_coord.append(env.quadrant_coords_list[i][0][0])
        y_coord.append(env.quadrant_coords_list[i][1][0])

    col_outline_protected = "black"
    lwd = 1

    fig_list = []
    fig_s = [6, 5.5]

    for sp_i in range(len(species_list)):
        fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
        mask = np.zeros(abundance_map[sp_i].shape)
        if log_transform:
            mask[abundance_map[sp_i] <= np.log10(2)] = 1
        else:
            mask[abundance_map[sp_i] < 1] = 1
        # cmap = sns.color_palette("crest")
        ax = sns.heatmap(
            abundance_map[sp_i],
            cmap="viridis_r",
            vmin=0,
            vmax=np.max(max_pop_sp),
            xticklabels=False,
            yticklabels=False,
            mask=mask,
        )
        ax.set_facecolor("#f0f0f0")
        if plot_titles:
            plt.gca().set_title(ttl[sp_i], fontweight="bold", fontsize=fontsize)
        for i in range(len(x_coord)):
            ax.add_patch(
                Rectangle(
                    (y_coord[i], x_coord[i]),
                    resolution[0],
                    resolution[1],
                    fill=False,
                    edgecolor=col_outline_protected,
                    lw=lwd,
                )
            )

        fig_list.append(fig)

    return fig_list, titles


def plot_biodiv_env(pklfile=None, loaded_env=None, max_n_species=0, plot_titles=True):
    if loaded_env:
        env = loaded_env
    else:
        with open(pklfile, "rb") as pkl:
            env = pickle.load(pkl)
    evolveGrid = env.bioDivGrid

    # ----
    fig_list = []
    titles = []
    fig_s = [6, 5.5]
    fontsize = 15
    col_outline_protected = "black"
    lwd = 1
    # get protected units
    q_indx = env.protected_quadrants
    x_coord, y_coord = [], []
    for i in q_indx:
        x_coord.append(env.quadrant_coords_list[i][0][0])
        y_coord.append(env.quadrant_coords_list[i][1][0])

    resolution = env.resolution
    time_series_stats = np.array(env.history) * 100
    # ----

    # plot sp richness
    titles.append("Species richness")
    ttl = "Species richness (%s ssp.)" % evolveGrid.numberOfSpecies()
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    if not max_n_species:
        bounds = [0, np.max(evolveGrid.speciesPerCell())]
    else:
        bounds = [0, max_n_species]
    ax = sns.heatmap(
        evolveGrid.speciesPerCell(),
        cmap="coolwarm",
        vmin=bounds[0],
        vmax=bounds[1],
        xticklabels=False,
        yticklabels=False,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # population density
    titles.append("Mean population density")
    ttl = "Mean population density (mean: %s)" % round(
        np.mean(evolveGrid.individualsPerCell()), 1
    )
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    bounds = [0, np.max(evolveGrid._K_max)]
    ax = sns.heatmap(
        evolveGrid.individualsPerCell(),
        cmap="YlGn",
        vmin=bounds[0],
        vmax=bounds[1],
        xticklabels=False,
        yticklabels=False,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # rank-abundance plot
    titles.append("Total population size")
    ttl = "Total population size (%s M)" % np.round(
        evolveGrid.numberOfIndividuals() / 1000000, 2
    )
    n_individuals_per_species = [
        np.arange(evolveGrid._n_species),
        evolveGrid.individualsPerSpecies(),
    ]
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # rank abundance plot
    plt.bar(
        x=n_individuals_per_species[0],
        height=n_individuals_per_species[1],
        width=0.8,
        linewidth=0,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    fig_list.append(fig)

    # phylogeny
    titles.append("Phylogenetic diversity")
    pd_percentage = np.round(time_series_stats[-1, 2], 1)
    ttl = "Phylogenetic diversity (%s %%)" % pd_percentage

    extant_species = evolveGrid._all_tip_labels[evolveGrid.extantSpeciesID()]
    extinct_species = [i for i in evolveGrid._all_tip_labels if i not in extant_species]
    try:
        ll = bt.loadNewick(evolveGrid._phylo_file_name, absoluteTime=False)
        # transform species name to match ll tree
        extinct_species = [str(int(i.split("T")[1])) for i in extinct_species]
    except:
        ll = convert_to_bt_tree(evolveGrid._phylo_tree)
    grey_out = [i for i in ll.getExternal() if i.name in extinct_species]
    for tip in grey_out:
        tip.traits["inactive"] = True  ## inactivate tips
    for node in ll.getInternal():  ## iterate over internal nodes
        if len(node.leaves) == len(
            [ch for ch in node.leaves if ch in [k.name for k in grey_out]]
        ):  ## if all descendant tips are grey'd out - grey out node too
            node.traits["inactive"] = True

    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    ax = fig.add_subplot(111, facecolor="w")  # phylogeny
    colour = (
        lambda k: "red" if "inactive" in k.traits else "darkgray"
    )  ## light grey if branch has "inactive" as key in trait dict, black otherwise
    ll.plotTree(
        ax, connection_type="elbow", width=1, colour=colour
    )  ## elbow branch connection, small branch width, colour via function
    ax.set_yticks([])
    ax.set_yticklabels([])  ## remove y axis labels
    ax.set_xticks([])
    ax.set_xticklabels([])  ## remove x axis labels
    [
        ax.spines[loc].set_visible(False) for loc in ax.spines if loc not in ["bottom"]
    ]  ## remove spines
    ax.set_xlim(-0.1, ll.treeHeight + 0.1)  ## limit tree
    ax.set_ylim(-2, ll.ySpan + 2)
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    fig_list.append(fig)

    # disturbance
    titles.append("Disturbance")
    ttl = "Disturbance (mean: %s)" % round(
        np.mean(evolveGrid._disturbance_matrix * (1 - evolveGrid._protection_matrix)), 2
    )
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    disturbance = evolveGrid._disturbance_matrix * (1 - evolveGrid._protection_matrix)
    ax = sns.heatmap(
        disturbance,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # selective disturbance
    titles.append("Selective disturbance")
    ttl = "Selective disturbance (mean: %s)" % round(
        np.mean(
            evolveGrid._selective_disturbance_matrix
            * (1 - evolveGrid._protection_matrix)
        ),
        2,
    )
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    selective_disturbance = evolveGrid._selective_disturbance_matrix * (
        1 - evolveGrid._protection_matrix
    )
    ax = sns.heatmap(
        selective_disturbance,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # climate
    titles.append("Mean annual temperature")
    ttl = "Mean annual temperature (mean anomaly: %s)" % round(
        np.mean(evolveGrid._climate_layer), 2
    )
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # climate
    ax = sns.heatmap(
        evolveGrid._climate_layer,
        cmap="Reds",
        vmin=0,
        vmax=7,
        xticklabels=False,
        yticklabels=False,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # value map
    titles.append("Economic loss")
    ttl = "Economic loss (%s %%)" % np.round(100 - time_series_stats[-1, 1], 1)
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))  # value map
    sp_value = evolveGrid._species_value_reference
    presence_absence = evolveGrid._h + 0
    presence_absence[
        presence_absence < 1
    ] = 0  # species_threshold is only used for total pop size
    presence_absence[presence_absence > 1] = 1  # not within each cell
    cell_value = np.log(1 + np.einsum("sij,s->ij", presence_absence, sp_value))
    bounds = [0, np.log((np.sum(sp_value)))]
    ax = sns.heatmap(
        cell_value, vmin=bounds[0], vmax=bounds[1], xticklabels=False, yticklabels=False
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        # print(i, y_coord[i], x_coord[i])
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # protection cost
    titles.append("Cost of protecting")
    costs = env.getProtectCostQuadrant()
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    cell_cost_matrix = np.zeros(env.bioDivGrid._protection_matrix.shape)
    for i in range(len(costs)):
        xy = np.meshgrid(env.quadrant_coords_list[i][0], env.quadrant_coords_list[i][1])
        cell_cost_matrix[xy[0], xy[1]] = costs[i]
    ttl = "Cost of protecting (mean: %s)" % round(
        np.mean(costs + env._baseline_cost), 2
    )

    max_cost = env._baseline_cost + env._cost_coeff * np.prod(env.resolution)
    ax = sns.heatmap(
        cell_cost_matrix,
        vmin=env._baseline_cost,
        vmax=max_cost,
        xticklabels=False,
        yticklabels=False,
    )
    if plot_titles:
        plt.gca().set_title(ttl, fontweight="bold", fontsize=fontsize)
    for i in range(len(x_coord)):
        ax.add_patch(
            Rectangle(
                (y_coord[i], x_coord[i]),
                resolution[0],
                resolution[1],
                fill=False,
                edgecolor=col_outline_protected,
                lw=lwd,
            )
        )
    fig_list.append(fig)

    # diversity/PD/value trajectories
    titles.append("Variables through time")
    ttl = "Variables through time"
    fig = plt.figure(figsize=(fig_s[0], fig_s[1]))
    x = np.arange(len(time_series_stats))
    plt.plot(x, time_series_stats[:, 1], "o-", color="#7570b3")  # value
    plt.plot(x, time_series_stats[:, 2], "o-", color="#1b9e77")  # PD
    plt.plot(x, time_series_stats[:, 0], "o-", color="#d95f02")  # sp num
    if plot_titles:
        plt.gca().set_title(
            "Variables through time", fontweight="bold", fontsize=fontsize
        )
    plt.legend(
        labels=("Economic value", "Phylogenetic diversity", "Species diversity"),
        loc="lower left",
    )
    plt.axis([-1, 31, 50, 105])  # TODO: expose xlim and ylim
    plt.xlabel("Time")
    plt.ylabel("Percentage values")
    fig_list.append(fig)

    return fig_list, titles


def _plot_env_state_init(env, species_list=None, plot_titles=True):
    if species_list == None:
        species_list = range(env.n_species)

    fig_list, titles = plot_biodiv_env(plot_titles=plot_titles, loaded_env=env)

    if len(species_list):
        # fig_list = fig_list + plot_species_ranges_list(
        species_figs, species_titles = plot_species_ranges_list(
            loaded_env=env,
            species_list=species_list,
            log_transform=1,
            plot_titles=plot_titles,
        )
        fig_list.extend(species_figs)
        titles.extend(species_titles)

    return fig_list, titles


def _plot_env_state_plot_fig(fig, outfile_name, wd, plot_count, title, file_format):
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    file_name = (
        os.path.join(wd, os.path.basename(outfile_name))
        + f"_p{plot_count} - {title}.{file_format}"
    )

    print(f"Save fig '{file_name}'")
    fig.savefig(file_name)
    return file_name


def plot_env_state(
    env,
    wd=".",
    outfile="sim",
    species_list=None,
    plot_titles=True,
    file_format="one_pdf",
):
    fig_list, titles = _plot_env_state_init(
        env=env, species_list=species_list, plot_titles=plot_titles
    )

    outfile_name = "%s_step_%s" % (outfile, env.currentIteration)

    if file_format == "one_pdf":
        file_name = os.path.join(wd, os.path.basename(outfile_name)) + ".pdf"
        plot_biodiv = matplotlib.backends.backend_pdf.PdfPages(file_name)

        for fig in fig_list:
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            plot_biodiv.savefig(fig)
        plot_biodiv.close()
        print("Plot saved as:", file_name)
        return

    if file_format not in ["pdf", "svg", "png", "jpg"]:
        file_format = "pdf"

    plot_count = 0
    for fig, title in zip(fig_list, titles):
        _plot_env_state_plot_fig(
            fig=fig,
            outfile_name=outfile_name,
            wd=wd,
            plot_count=plot_count,
            title=title,
            file_format=file_format,
        )
        plot_count += 1


def plot_env_state_generator(
    env,
    wd=".",
    outfile="sim",
    species_list=None,
):
    fig_list, titles = _plot_env_state_init(
        env=env, species_list=species_list, plot_titles=True
    )

    outfile_name = "%s_step_%s" % (outfile, env.currentIteration)

    file_format = "svg"

    plot_count = 0
    for fig, title in zip(fig_list, titles):
        filename = _plot_env_state_plot_fig(
            fig=fig,
            outfile_name=outfile_name,
            wd=wd,
            plot_count=plot_count,
            title=title,
            file_format=file_format,
        )

        yield {
            "type": "plot",
            "status": "progress",
            "data": {
                "step": env.currentIteration,
                "plot": plot_count,
                "num_plots": len(fig_list),
                "filename": filename,
                "title": title,
            },
        }

        plot_count += 1
