import numpy as np
from buffeting import beta_and_theta_bar_func, beta_0_func
from simple_5km_bridge_geometry import g_node_coor
from transformations import T_LsGs_3g_func, T_GsGw_func
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

n_samples = 1000000  #  10000000
n_nodes_plot = 3
beta_DB = rad(100-40)  #  rad(270+45+22.5)
beta_0 = beta_0_func(beta_DB)
theta_0 = rad(0)
n_g_nodes = len(g_node_coor)

# Mean wind-speed
z = 14.5
RP = 100
z0 = 0.01
kt = 0.17
K = 0.2
n = 0.5
p = 1 - np.exp(-1 / RP)
Cr = kt * np.log(z / z0)
Cprob = ((1 - K * np.log(-np.log(1 - p))) / (1 - K * np.log(-np.log(0.98)))) ** n
V_1hour = Cr * Cprob * 24.3
V_10min = V_1hour * 1.07
U_bar = V_10min

# Turbulence
Iu = 0.137
Iv = 0.115
Iw = 0.082
sigma_u = Iu * U_bar
sigma_v = Iv * U_bar
sigma_w = Iw * U_bar

# This is not the same as real wind time series. These assume totally uncorrelated occurrences of wind speeds (and u and w have a known correlation)
u = np.random.normal(0, sigma_u, n_samples)
v = np.random.normal(0, sigma_v, n_samples)
w = np.random.normal(0, sigma_w, n_samples)
V = np.sqrt((U_bar+u)**2 + v**2 + w**2)

# Finding beta_bar and theta_bar at the selected nodes
n_g_elems = n_g_nodes - 1
n_equidist_elems_plot = n_nodes_plot - 1
idx_nodes_plot = np.arange(n_nodes_plot) * int(n_g_elems / n_equidist_elems_plot)
beta_bar_all_nodes, theta_bar_all_nodes = beta_and_theta_bar_func(g_node_coor, beta_0=beta_0, theta_0=theta_0, alpha=None)  # you need to put all g_node_coor here! Because angles are calculated from an approximate polygon-shaped bridge
T_LsGs_all_nodes = T_LsGs_3g_func(g_node_coor, alpha=None)

# Focusing only on the selected nodes to be plotted (for computer efficiency)
g_node_coor_plot = g_node_coor[idx_nodes_plot]
beta_bar_plot, theta_bar_plot = beta_bar_all_nodes[idx_nodes_plot], theta_bar_all_nodes[idx_nodes_plot]
U_bar_plot = U_bar*np.ones(n_nodes_plot)
T_LsGs_plot = T_LsGs_all_nodes[idx_nodes_plot]
T_GsGw = T_GsGw_func(beta_0, theta_0)
T_LsGw_plot = np.einsum('nij,jk->nik', T_LsGs_plot, T_GsGw, optimize=True)
U_Gw_tilde = np.array([U_bar_plot[:,None]+u, np.zeros(n_nodes_plot)[:,None]+v, np.zeros(n_nodes_plot)[:,None]+w])  # shape (3,n,s), where s -> number of samples
U_Ls_tilde = np.einsum('nij,jns->ins', T_LsGw_plot, U_Gw_tilde, optimize=True)
U_tilde = np.sqrt(U_Ls_tilde[0]**2+U_Ls_tilde[1]**2+U_Ls_tilde[2]**2)
U_xy_tilde = np.sqrt(U_Ls_tilde[0]**2+U_Ls_tilde[1]**2)


beta_tilde = -np.arccos(U_Ls_tilde[1] / U_xy_tilde) * np.sign(U_Ls_tilde[0])
theta_tilde = np.arcsin(U_Ls_tilde[2] / U_tilde)
theta_yz_tilde = np.arcsin(np.sin(theta_tilde) / np.sqrt(1-np.sin(beta_tilde)**2 * np.cos(theta_tilde)**2))

beta_tilde_deg = np.rad2deg(beta_tilde)
theta_tilde_deg = np.rad2deg(theta_tilde)
theta_yz_tilde_deg = np.rad2deg(theta_yz_tilde)

# Now, one node at the time
pt_indxs_plot = [0,1,2]




df =           pd.DataFrame({'type': [r'$\tilde{\theta}_{yz}$']*n_samples})
df = df.append(pd.DataFrame({'type': [r'$\tilde{\theta}$']     *n_samples}))
df = pd.DataFrame()
for i in pt_indxs_plot:
    df = df.append(pd.DataFrame({'point':i, 'type': [r'$\tilde{\theta}_{yz}$'] * n_samples, f'beta': beta_tilde_deg[i], f'angle_P{i}': theta_yz_tilde_deg[i]}), ignore_index=True)
    df = df.append(pd.DataFrame({'point':i, 'type': [r'$\tilde{\theta}$']      * n_samples, f'beta': beta_tilde_deg[i], f'angle_P{i}':    theta_tilde_deg[i]}), ignore_index=True)



# # ATTEMPTING MYSELF
# df = pd.DataFrame()
# for i in pt_indxs_plot:
#     df = pd.concat([df, pd.DataFrame({f'beta_P{i}':beta_tilde_deg[i], f'theta_P{i}':theta_tilde_deg[i], f'theta_yz_P{i}':theta_yz_tilde_deg[i]})], axis=1)
#
# for i in pt_indxs_plot:
#     sns.set(font_scale=1.6)
#     sns.set_style("whitegrid")
#     # Scatter plot
#     g = sns.jointplot(x=f"beta_P{i}", y=f"theta_P{i}", data=df, kind='scatter', alpha=0.2, ci=None, joint_kws={"s": 4})
#     plt.show()
#
#
#     # KDE plot
#     g = sns.jointplot(x=r"beta", y="angle", data=df, levels=6, kind='kde', hue='type', ci=None)
#     g.ax_joint.set(xlabel=r'$\beta$  [deg]', ylabel=r'$\theta$  or  $\theta_{yz}$  [deg]')
#     g.ax_marg_x.set(xlim=[60,180])
#     g.ax_marg_y.set(ylim=[-90, 90])
#     g.ax_joint.legend_.set_title(None)
#     plt.show()



# for i in pt_indxs_plot:
#     sns.set(font_scale=1.6)
#     sns.set_style("whitegrid")
#     # Scatter plot
#     g = sns.jointplot(x=r"beta", y="angle", data=df, kind='scatter', hue='type', alpha=0.2, ci=None, joint_kws={"s": 4})
#     # KDE plot
#     g = sns.jointplot(x=r"beta", y="angle", data=df, levels=6, kind='kde', hue='type', ci=None)
#     g.ax_joint.set(xlabel=r'$\beta$  [deg]', ylabel=r'$\theta$  or  $\theta_{yz}$  [deg]')
#     g.ax_marg_x.set(xlim=[60,180])
#     g.ax_marg_y.set(ylim=[-90, 90])
#     g.ax_joint.legend_.set_title(None)
#     plt.show()








# ATTEMPT TO REPLICATE https://jehyunlee.github.io/2020/10/03/Python-DS-35-seaborn_matplotlib2/
def jointplots(x, ys, data, hue=None, hue2=None, width=6, ratio=5, space=0.2, markeralpha=0.3, markersize=2, xlabel=None, ylabels=None, supylabel=None, xlims=[-180,180], ylims=[-90,90], margin_norm=False):
    """
    -------------------
    Input Parameters
    -------------------
    xs      : (list or str) feature name(s) of data
    y       : (str) feature name of data
    data    : (pandas.DataFrame)
    hue     : (str) semantic variable that is mapped to determine the color of plot elements. Semantic variable that is mapped to determine the color of plot elements.

    height  : (float) size of the figure
    ratio   : (float) ratio of the joint (main) axes height to marginal axes height.
    space   : (float) space between the joint and marginal axes

    xlabels : (list or str) xlabels
    ylabel  : (str) ylabel
    margin_norm : (boolean) if True, kdeplots at marginal axes have same scale.
    """
    ### 1. input check
    # input type
    assert isinstance(ys, list) or isinstance(ys, str)
    if isinstance(ys, list):
        assert all([isinstance(y, str) for y in ys])
    else:
        ys = [ys]

    if ylabels != None:
        assert isinstance(ylabels, list) or isinstance(ylabels, str)
        if isinstance(ylabels, list):
            assert all([isinstance(ylabel, str) for ylabel in ylabels])
        else:
            ylabels = [ylabels]

    if xlabel != None:
        assert isinstance(xlabel, str)

    if hue != None:
        assert isinstance(hue, str)

    # input data
    assert all([y in data.columns for y in ys])
    assert x in data.columns
    if hue != None:
        assert hue in data.columns

    ### 2. figure
    w_margin = width / (ratio + 1)
    w_joint = width - w_margin

    if isinstance(ys, list):
        n_y = len(ys)
    else:
        n_y = 1

    heights = list(np.array([w_margin] + [w_joint] * n_y) * 0.69)
    widths = [w_joint, w_margin]
    nrows = len(heights)
    ncols = len(widths)

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set_palette([sns.color_palette()[i] for i in [2,1,3,4,5,6,7]])
    fig = plt.figure(figsize=(sum(widths)+w_margin, sum(heights)-w_margin), constrained_layout=True, dpi=300)

    ### 3. gridspec preparation
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows,
                            width_ratios=widths, height_ratios=heights,
                            wspace=space, hspace=space)

    ### 4. setting axes
    axs = {}
    for i in range(ncols * nrows):
        axs[i] = fig.add_subplot(spec[i // ncols, i % ncols])

    axes_j = 2*np.arange(1, nrows)  # indexes of the "joint" axes where the main plots are

    ### 5. jointplots (scatterplot + kdeplot)
    for i, y in zip(axes_j, ys):
        # Getting percentiles:
        for j,a in enumerate(df[hue].unique()):  # iterating through each hue type (e.g. theta, and theta_yz)
            data_1_point_1_angle_type = df[df[hue] == a][y].dropna()
            p10 = np.percentile(data_1_point_1_angle_type, 10)
            p90 = np.percentile(data_1_point_1_angle_type, 90)
            axs[i].hlines(y=[p10,p90], xmin=xlims[0], xmax=xlims[1],  linestyles="--", linewidth=2, alpha=0.5, colors=sns.color_palette()[j])
            # axs[i].text(x=xlims[0], y=p90*0.9, s='P90', c=sns.color_palette()[j])
        axs[i].axvline(x=90, linestyle="-.", linewidth=2., alpha=0.5, color='black')
        if i == axes_j[0]:  # if first plot
            legend = True
        else:
            legend = False
        sns.kdeplot(x=x, y=y, data=data, hue=hue, alpha=0.5, levels=np.linspace(0.1,0.9,9), ax=axs[i], zorder=3, legend=legend)
        # sns.scatterplot(x=x, y=y, data=data, hue=hue, alpha=markeralpha, ax=axs[i], s=markersize, zorder=2, legend=legend)
        sns.histplot(x=x, y=y, data=data, hue=hue, ax=axs[i], zorder=2, legend=legend, alpha=0.4)
        if i == axes_j[0]:  # if first plot (and it has legend)
            axs[i].legend_.set_title('')
            axs[i].get_legend()._loc = 1
            plt.setp(axs[i].get_legend().get_texts(), fontsize='25')  # for legend title
            # test.legend(loc=1)  # for legend title
            for lh in axs[i].get_legend().legendHandles:
                lh.set_alpha(0.5)
        axs[i].set_xlim(xlims)
        axs[i].set_ylim(ylims)
        axs[i].set_xticks([-30,-15,0,15,30,45,60,75,90,105,120])   #([75, 90, 105, 120, 135, 150, 165])
        axs[i].set_yticks([-45,-30,-15,0,15,30,45])
        axs[i].grid("on", color="lightgray", zorder=0)
        axs[i].tick_params(labelsize=25)



    ### 6. kdeplots at marginal axes
    axs[ncols - 1].axis("off")

    axes_mx = 0  # index of the first horizontal marginal axis
    axes_my = list(axes_j + 1)   # index of the vertical marginal axis

    # Several vertical marginal plots
    for i, j, y in zip(axes_my,axes_j, ys):
        sns.kdeplot(y=y, data=data, hue=hue, fill=True, ax=axs[i], zorder=2, legend=False)
        axs[i].set_ylim(ylims)
        axs[i].set_ylabel("")
        axs[i].set_yticklabels([])
        axs[i].spines["left"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

    # One horizontal marginal plot
    color_top_marginal_plot = sns.color_palette()[6]  # 5: brown - ish (or 6, if one color was already removed from the palette with sns.set_palette)
    sns.kdeplot(x=x, data=data, hue=hue2, fill=True, ax=axs[axes_mx], zorder=2, legend=False, palette=[color_top_marginal_plot]*len(data[hue2].unique()))  # color of the top horizontal marginal plot. Choose number from here https://seaborn.pydata.org/tutorial/color_palettes.html
    axs[axes_mx].set_xlim(xlims)
    axs[axes_mx].set_xlabel("")
    axs[axes_mx].set_xticklabels([])
    axs[axes_mx].spines["bottom"].set_visible(False)
    axs[axes_mx].spines["top"].set_visible(False)
    axs[axes_mx].spines["right"].set_visible(False)

    if margin_norm == True:
        hist_range_max = max([axs[m].get_xlim()[-1] for m in axes_my] + [axs[axes_mx].get_ylim()[-1]])
        for i in axes_my:
            axs[i].set_xlim(0, hist_range_max)
        axs[axes_mx].set_ylim(0, hist_range_max)

    ### 7. unnecessary elements removal
    # 7.1. labels and ticklabels
    for i in axes_j[:-1]:
        axs[i].set_xlabel("")
        axs[i].set_xticklabels([])

    # 7.2. marginal axes
    for i in axes_my:
        axs[i].grid(False) # , color="lightgray", zorder=0)
        axs[i].set_yticklabels([])
        # if i != axes_my[-1]:
        axs[i].set_xlabel("")  # Removes "Density" label from hidden places
        axs[i].set_xticklabels([])
    axs[axes_mx].grid(False) # , color="lightgray", zorder=0)
    axs[axes_mx].set_xticklabels([])
    axs[axes_mx].set_yticklabels([])
    # Remove ticks
    for i in axes_my+[axes_mx]:
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set(frame_on=False)

    # 7.3. labels
    font_label = {"color": "black", "fontsize":"large"}
    labelpad = 12
    for i, y in zip(axes_j, ylabels):
        axs[i].set_ylabel(y, fontdict=font_label, labelpad=labelpad, color='darkgrey')
        if i == axes_j[-1]:
            axs[i].set_xlabel(xlabel, fontdict=font_label, labelpad=labelpad)
    axs[0].set_ylabel("", fontdict=font_label, labelpad=labelpad)
    axs[2*nrows-1].set_xlabel("", fontdict=font_label, labelpad=labelpad)

    fig.supylabel(supylabel)
    # plt.tight_layout()

    return fig, axs

jointplots(ys=["angle_P2", "angle_P1", "angle_P0"], x="beta", data=df, hue="type", hue2="point",
            width=8, ratio=5, space=0.03, markeralpha=0.2, markersize=3,
            xlabel=r'$\tilde{\beta}$  [deg]',
            ylabels=[r"North end", r"Mid-span", r"South end"],
            supylabel=r'$\tilde{\theta}$  or  $\tilde{\theta}_{yz}$  [deg]',
            xlims=[-30,120], ylims=[-45,45], margin_norm=False)
plt.savefig('plots/betas_and_thetas_PDFs_5.png')
plt.show()












# df_1pt = pd.DataFrame()
# df_1pt = df_1pt.append(pd.DataFrame({'point':i, 'beta':beta_tilde_deg[i], 'angle':theta_yz_tilde_deg[i]}))
# df_1pt = df_1pt.append(pd.DataFrame({'point':i, 'beta':beta_tilde_deg[i], 'angle':   theta_tilde_deg[i]}))
# df = pd.concat([df, df_1pt], axis=1)







# penguins = sns.load_dataset( "penguins" )
# jointplots(["bill_length_mm", "bill_depth_mm", "flipper_length_mm"], "body_mass_g", penguins, hue="species",
#             height=8, ratio=5, space=0.03,
#             xlabels=["Bill Length (mm)", "Bill Depth (mm)", "Flipper Length (mm)"], ylabel="Body Mass (g)")
#






