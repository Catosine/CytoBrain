# Made by Cyto
#     　　　　 ＿ ＿
# 　　　　　／＞　　 フ
# 　　　　　|   _　 _l
# 　 　　　／` ミ＿xノ
# 　　 　 /　　　 　 |
# 　　　 /　 ヽ　　 ﾉ
# 　 　 │　　|　|　|
# 　／￣|　　 |　|　|
# 　| (￣ヽ＿_ヽ_)__)
# 　＼二つ ；
from time import strftime
import os
import os.path as osp

import numpy as np
import pandas as pd
import plotly.express as px


def classiy_correlation_by_roi(roi_path, lh_correlation, rh_correlation):
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(osp.join(roi_path, 'roi_masks', r),
                                     allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
                              'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
                              'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
                              'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
                              'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
                              'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
                              'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for l, r in zip(lh_challenge_roi_files, rh_challenge_roi_files):
        lh_challenge_rois.append(
            np.load(osp.join(roi_path, 'roi_masks', l)))
        rh_challenge_rois.append(
            np.load(osp.join(roi_path, 'roi_masks', r)))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0:  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    return roi_names, lh_roi_correlation, rh_roi_correlation


def histogram(roi_path, lh_correlation, rh_correlation, title, save=None):
    """
        Visualize the correlation result
        Args:
            roi_path,           str, path to ROI masks
            lh_correlation,     np.ndarray, left hemisphere correlation
            rh_correlation,     np.ndarray, right hemishphere correlation
            title,              str, title of the figure
            save,               str, where to save the graph
    """

    roi_names, lh_roi_correlation, rh_roi_correlation = classiy_correlation_by_roi(
        roi_path, lh_correlation, rh_correlation)

    lh_median_roi_correlation = [
        np.median(corr) for corr in lh_roi_correlation]
    rh_median_roi_correlation = [
        np.median(corr) for corr in rh_roi_correlation]

    df = pd.DataFrame({"ROIs": roi_names + roi_names, "Median Pearson's R": lh_median_roi_correlation + rh_median_roi_correlation,
                       "Hemisphere": ["Left"] * len(lh_roi_correlation) + ["Right"] * len(rh_roi_correlation)})
    # draw the diagram
    fig = px.histogram(df, x="ROIs", y="Median Pearson's R", color="Hemisphere",
                       hover_data=df.columns.tolist(), barmode="group", width=1500, height=500)
    fig.update_xaxes(categoryorder='array',
                     categoryarray=roi_names, tickangle=45)
    fig.update_layout(title_text=title, yaxis=dict(range=[0.0, 1.0]))

    if save:

        if not osp.isdir(save):
            os.makedirs(save)

        to_save = osp.join(save, "histogram_pearson_{}".format(
            strftime("%Y%m%d%H%M%S")))

        fig.write_html(to_save+".html")
        fig.write_image(to_save+".png")

    return fig


def box_plot(roi_path, lh_correlation, rh_correlation, title, save=None):
    """
        Visualize the correlation result
        Args:
            roi_path,           str, path to ROI masks
            lh_correlation,     np.ndarray, left hemisphere correlation
            rh_correlation,     np.ndarray, right hemishphere correlation
            title,              str, title of the figure
            save,               str, where to save the graph
    """

    roi_names, lh_roi_correlation, rh_roi_correlation = classiy_correlation_by_roi(
        roi_path, lh_correlation, rh_correlation)

    r = list()
    p = list()
    h = list()
    for i in range(len(roi_names)):
        r += [roi_names[i]] * len(lh_roi_correlation[i])
        p += lh_roi_correlation[i].tolist()
        h += ["Left"] * len(lh_roi_correlation[i])

        r += [roi_names[i]] * len(rh_roi_correlation[i])
        p += rh_roi_correlation[i].tolist()
        h += ["Right"] * len(rh_roi_correlation[i])

    df = pd.DataFrame({"ROIs": r, "Pearson's R": p, "Hemisphere": h})

    # draw the diagram
    fig = px.box(df, x="ROIs", y="Pearson's R", color="Hemisphere",
                 hover_data=df.columns.tolist(), width=1500, height=500)
    fig.update_xaxes(categoryorder='array',
                     categoryarray=roi_names, tickangle=45)
    fig.update_layout(title_text=title)

    if save:

        if not osp.isdir(save):
            os.makedirs(save)

        to_save = osp.join(save, "box_pearson_{}".format(
            strftime("%Y%m%d%H%M%S")))

        fig.write_html(to_save+".html")
        fig.write_image(to_save+".png")

    return fig


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        help="path to data set where keeps ROI masks")
    parser.add_argument("--l_correlation", type=str,
                        help="Correlation of left hemishpere")
    parser.add_argument("--r_correlation", type=str,
                        help="Correlation of right hemishpere")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--title", type=str,
                        help="Title of the generated figs")

    args = parser.parse_args()

    l_correlation = np.load(args.l_correlation)
    r_correlation = np.load(args.r_correlation)

    histogram(args.data, l_correlation, r_correlation,
              args.title, args.save_path)
    box_plot(args.data, l_correlation, r_correlation,
             args.title, args.save_path)
