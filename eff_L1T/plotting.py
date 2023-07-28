import os
import time

import pickle
import hist
from hist.intervals import clopper_pearson_interval
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
from matplotlib.colors import LogNorm
mpl.rcParams.update({'font.size': 20})
mpl.use("Agg")
mplhep.style.use("CMS")

## Parameters that are changeable in main script
from config import FULL_TAG, LOGX_PT

## Parameters that are fixed by config
from config import BINS_DICT, LABELS, trig_WP, trig_PT


def plotting(
        outpath, RUN, files_key, full_tag,
        FULL_TAG=FULL_TAG, LOGX_PT=LOGX_PT,
):
    local_time = time.time()
    print(f".... In plotting: Creating plots for tag {full_tag}")
    TFs = ["KBMTf", "OMTf", "EMTf"]
    if RUN == 2:
        TFs = ["BMTf", "OMTf", "EMTf"]

    prb_pt = full_tag.split("_")[1].replace("prb", "")
    tag_pt = full_tag.split("_")[0].replace("tag", "")

    TF_labels = {
        "uGMT": r"$0.00 < \eta < 2.40$",
        "BMTf": r"$0.00 < \eta < 0.83$",
        "KBMTf": r"$0.00 < \eta < 0.83$",
        "OMTf": r"$0.83 < \eta < 1.24$",
        "EMTf": r"$1.24 < \eta < 2.40$",
    }
    cms_label_kwargs = {"label": files_key, "fontsize": 22, "data": True}
    if RUN == 3: cms_label_kwargs["rlabel"] = "(13.6 TeV)"

    ## create required folders
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    if not os.path.isdir(outpath+"/png"):
        os.mkdir(outpath+"/png")
    if not os.path.isdir(outpath+"/pdf"):
        os.mkdir(outpath+"/pdf")

    ## load histograms from pickle
    with open(f"{outpath}/histograms.pickle", "rb") as handle:
        histograms = pickle.load(handle)

    ## Plotting functions are called at the bottom

    def plot_eff_2d(histograms):
        for k in trig_WP.keys():
            fig, ax = plt.subplots()

            axes = histograms["trg_eta_vs_phi"].axes
            h_num = histograms["trg_eta_vs_phi"][{"trig_WP": k, "TF": "uGMT"}]
            h_denom = histograms["eta_vs_phi"][{"TF": "uGMT"}]

            # rebin
            #h_num = h_num[{"probe_eta": hist.rebin(2), "probe_phi": hist.rebin(2)}]
            #h_denom = h_denom[{"probe_eta": hist.rebin(2), "probe_phi": hist.rebin(2)}]

            vals = h_num / h_denom
            vals.plot2d(ax=ax)

            mplhep.cms.label(ax=ax, **cms_label_kwargs)

            if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)
            plt.annotate(r"$p_{T}^{L1} >$" + str(trig_PT[k]) + r", $L1_{qual} \geq$" + str(trig_WP[k][0]), (40, 510), xycoords="axes points", color="black", fontsize=24)
            plt.tight_layout()

            fig.savefig(f"{outpath}/pdf/eff_eta_vs_phi_{k}.pdf")
            fig.savefig(f"{outpath}/png/eff_eta_vs_phi_{k}.png")
            plt.close()



    def plot_eff_TFs(histograms):
        print("Create efficiency plots for all variables and all trig WPs (overlaying eta regions)....")
        for var, bins in BINS_DICT.items():
            for k in trig_WP.keys():
                #if k != "SingleMu" or var !="pt": continue

                fig, ax = plt.subplots()
                axes = histograms["trg_"+var][{"trig_WP": k}].axes

                for i, iTF in enumerate(axes["TF"]):
                    h_num = histograms["trg_"+var][{"trig_WP": k, "TF": iTF}].values()
                    h_denom = histograms[var][{"TF": iTF}].values()
                    vals = np.divide(h_num, h_denom, out=np.zeros_like(h_num), where=h_denom!=0)
                    yerr = clopper_pearson_interval(h_num, h_denom)
                    yerr = np.where(vals==0, 0, np.abs(yerr - vals))
                    xerr = (axes["probe_"+var].edges[1:] - axes["probe_"+var].edges[:-1]) / 2
                    plot_kwargs = {
                        "x": axes["probe_"+var].centers, "y": vals,
                        "yerr": yerr,
                        "xerr": xerr,
                        "label": TF_labels[iTF],  ## alternatively: iTF
                        "linestyle": "none", "marker": "D",
                        "markersize": 7, "elinewidth": 2,
                        "capsize": 3,
                    }
                    ax.errorbar(**plot_kwargs)

                ax.grid(color='grey', linestyle='--', linewidth=0.5)
                ax.set_yticks(ticks=np.arange(0, 1.51, 0.1))
                ax.set(xlim=(BINS_DICT[var][0], BINS_DICT[var][-1]), ylim=(0.0, 1.01), ylabel="L1T Efficiency", xlabel = LABELS[var])
                #ax.set(ylim=(0.0, 1.4))  # for efficiency ratio
                if (var=="pt") & LOGX_PT: ax.set(xlim=(1, BINS_DICT[var][-1]), xscale="log")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

                mplhep.cms.label(ax=ax, **cms_label_kwargs)
                leg_title = r"$p_{T}^{L1} >$" + str(trig_PT[k]) + r", $L1_{qual} \geq$" + str(trig_WP[k][0])
                plt.legend(loc="best", title=leg_title, fontsize=20, title_fontsize=22)
                plt.tight_layout()

                fig.savefig(f"{outpath}/pdf/eff_{var}_{k}.pdf")
                fig.savefig(f"{outpath}/png/eff_{var}_{k}.png")
                plt.close()

    def plot_eff_trigWPs(histograms):
        print("Create efficiency plots for all variables and all trig WPs (overlaying Trigger WP's)....")
        for var, bins in BINS_DICT.items():
            for iTF in TFs+["uGMT"]:
                fig, ax = plt.subplots()
                axes = histograms["trg_"+var][{"TF": iTF}].axes

                for k in trig_WP.keys():
                    h_num = histograms["trg_"+var][{"trig_WP": k, "TF": iTF}].values()
                    h_denom = histograms[var][{"TF": iTF}].values()
                    vals = np.divide(h_num, h_denom, out=np.zeros_like(h_num), where=h_denom!=0)
                    yerr = clopper_pearson_interval(h_num, h_denom)
                    yerr = np.where(vals==0, 0, np.abs(yerr - vals))
                    xerr = (axes["probe_"+var].edges[1:] - axes["probe_"+var].edges[:-1]) / 2
                    plot_kwargs = {
                        "x": axes["probe_"+var].centers, "y": vals,
                        "yerr": yerr,
                        "xerr": xerr,
                        "label": r"$p_{T}^{L1} >$" + str(trig_PT[k]) + r", $L1_{qual} \geq$" + str(trig_WP[k][0]),
                        "linestyle": "none", "marker": "D",
                        "markersize": 7, "elinewidth": 2,
                        "capsize": 3,
                    }
                    ax.errorbar(**plot_kwargs)

                ax.grid(color='grey', linestyle='--', linewidth=0.5)
                ax.set_yticks(ticks=np.arange(0, 1.51, 0.1))
                ax.set(xlim=(BINS_DICT[var][0], BINS_DICT[var][-1]), ylim=(0.0, 1.01), ylabel="L1T Efficiency", xlabel = LABELS[var])
                #ax.set(ylim=(0.0, 1.4))  # for efficiency ratio
                if (var=="pt") & LOGX_PT: ax.set(xlim=(1, BINS_DICT[var][-1]), xscale="log")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

                mplhep.cms.label(ax=ax, **cms_label_kwargs)
                plt.legend(loc="best", title=TF_labels[iTF], fontsize=20, title_fontsize=24)
                plt.tight_layout()
                fig.savefig(f"{outpath}/pdf/eff_{var}_{iTF}.pdf")
                fig.savefig(f"{outpath}/png/eff_{var}_{iTF}.png")
                plt.close()


    def plot_trig_probe_eta_vs_phi(histograms):
        print("Create 2d plots for triggered probe muons")
        for iTF in ["uGMT"]:  # +TFs
            for k in trig_WP.keys():
                fig, ax = plt.subplots()

                histograms["trg_eta_vs_phi"][{"TF": iTF, "trig_WP": k}].plot2d(ax=ax)

                mplhep.cms.label(ax=ax, **cms_label_kwargs)

                plt.annotate(TF_labels[iTF], (0.2, 0.82), xycoords="figure fraction", color="grey")
                plt.annotate(r"$p_{T}^{probe} >$" + str(prb_pt) + ", $p_{T}^{tag} >$" + str(tag_pt), (0.2, 0.76), xycoords="figure fraction", color="grey")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

                plt.tight_layout()

                fig.savefig(f"{outpath}/pdf/trig_probe_eta_vs_phi_{iTF}_{k}.pdf")
                fig.savefig(f"{outpath}/png/trig_probe_eta_vs_phi_{iTF}_{k}.png")



    def plot_probe_vs_tf_dR(histograms):
        print("Create probe var vs dR(Probe, TfMatch) plots")
        for var, bins in BINS_DICT.items():
            for iTF in ["uGMT"]: # +TFs
                fig, ax = plt.subplots()
                histograms["probe_"+var+"_vs_tf_dR"][{"TF": iTF}].plot2d(ax=ax, norm=LogNorm())

                mplhep.cms.label(ax=ax, **cms_label_kwargs)

                plt.annotate(TF_labels[iTF], (0.2, 0.82), xycoords="figure fraction", color="grey")
                plt.annotate(r"$p_{T}^{probe} >$" + str(prb_pt) + ", $p_{T}^{tag} >$" + str(tag_pt), (0.2, 0.76), xycoords="figure fraction", color="grey")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

                plt.tight_layout()

                fig.savefig(f"{outpath}/pdf/probe_{var}_vs_tf_dR_{iTF}.pdf")
                fig.savefig(f"{outpath}/png/probe_{var}_vs_tf_dR_{iTF}.png")
                plt.close()


    def plot_probe_vs_tfmatch(histograms):
        print("Create probe vs tfmatch plots")
        for var, bins in BINS_DICT.items():
            for iTF in ["uGMT"]: # +TFs
                fig, ax = plt.subplots()
                histograms["probe_vs_tfmatch_"+var][{"TF": iTF}].plot2d(ax=ax, norm=LogNorm())

                mplhep.cms.label(ax=ax, **cms_label_kwargs)

                plt.annotate(TF_labels[iTF], (0.2, 0.82), xycoords="figure fraction", color="grey")
                plt.annotate(r"$p_{T}^{probe} >$" + str(prb_pt) + ", $p_{T}^{tag} >$" + str(tag_pt), (0.2, 0.76), xycoords="figure fraction", color="grey")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

                plt.tight_layout()

                fig.savefig(f"{outpath}/pdf/probe_vs_tfmatch_{var}_{iTF}.pdf")
                fig.savefig(f"{outpath}/png/probe_vs_tfmatch{var}_{iTF}.png")
                plt.close()


    def plot_tag_and_probe(histograms):
        print("Create nTags vs nProbes plot....")
        fig, ax = plt.subplots()
        histograms["nTags_vs_nProbes"].plot2d(ax=ax)
        mplhep.cms.label(ax=ax, **cms_label_kwargs)
        plt.tight_layout()
        fig.savefig(f"{outpath}/pdf/nTags_vs_nProbes.pdf")
        fig.savefig(f"{outpath}/png/nTags_vs_nProbes.png")

        print("Create M_inv plot....")
        fig, ax = plt.subplots()
        histograms["tag_probe_mass"].plot1d(ax=ax, histtype="step")
        ax.set(ylabel="Number of tag-probe pairs")
        mplhep.cms.label(ax=ax, **cms_label_kwargs)
        plt.tight_layout()
        fig.savefig(f"{outpath}/pdf/m_tag_probe.pdf")
        fig.savefig(f"{outpath}/png/m_tag_probe.png")

        print("Create plots for denominator distributions....")
        for var, bins in BINS_DICT.items():
            fig, ax = plt.subplots()

            histograms[var].plot1d(ax=ax, overlay="TF", histtype="step")

            ax.set(xlim=(BINS_DICT[var][0], BINS_DICT[var][-1]), ylabel="Number of Probe Muons")
            if (var=="pt") & LOGX_PT: ax.set(xlim=(1, BINS_DICT[var][-1]), xscale="log", yscale="log")
            if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

            mplhep.cms.label(ax=ax, **cms_label_kwargs)

            plt.legend(loc="best", title=r"$p_{T}^{probe} >$" + str(prb_pt), fontsize=20)
            plt.tight_layout()

            fig.savefig(f"{outpath}/pdf/probe_{var}.pdf")
            fig.savefig(f"{outpath}/png/probe_{var}.png")
            plt.close()

        print("Create plots for Tag Muon....")
        for var, bins in BINS_DICT.items():
            fig, ax = plt.subplots()

            histograms["tag_"+var].plot1d(ax=ax, overlay="TF", histtype="step")

            ax.set(xlim=(BINS_DICT[var][0], BINS_DICT[var][-1]), ylabel="Number of Tag Muons")
            if (var=="pt") & LOGX_PT: ax.set(xlim=(1, BINS_DICT[var][-1]), xscale="log", yscale="log")
            if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

            mplhep.cms.label(ax=ax, **cms_label_kwargs)

            plt.legend(loc="best", title=r"$p_{T}^{tag} >$" + str(tag_pt), fontsize=20)
            plt.tight_layout()

            fig.savefig(f"{outpath}/pdf/tag_{var}.pdf")
            fig.savefig(f"{outpath}/png/tag_{var}.png")
            plt.close()


    def plot_tag_vs_probe(histograms):
        print("Create 2D Plots for Tag vs Probe Reco Muons....")
        for var, bins in BINS_DICT.items():
            for iTF in ["uGMT"]: # +TFs
                fig, ax = plt.subplots()
                histograms["tag_vs_probe_"+var][{"TF": iTF}].plot2d(ax=ax, norm=LogNorm())

                mplhep.cms.label(ax=ax, **cms_label_kwargs)

                plt.annotate(TF_labels[iTF], (0.2, 0.82), xycoords="figure fraction", color="grey")
                plt.annotate(r"$p_{T}^{probe} >$" + str(prb_pt) + ", $p_{T}^{tag} >$" + str(tag_pt), (0.2, 0.76), xycoords="figure fraction", color="grey")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)

                plt.tight_layout()

                fig.savefig(f"{outpath}/pdf/tag_vs_probe_{var}_{iTF}.pdf")
                fig.savefig(f"{outpath}/png/tag_vs_probe_{var}_{iTF}.png")
                plt.close()


    def plot_tag_vs_tf(histograms):
        print("Create 2D Plots for the eta/phi of reco vs iTF muon....")
        ## Could also produce these plots for each trig_WP, but naaah
        for var, bins in BINS_DICT.items():
            if var == "pt": continue
            for iTF in ["uGMT"]: # +TFs
                fig, ax = plt.subplots()
                histograms["trg_"+var+"_vs_PF"][{"trig_WP": "SingleMu", "TF": iTF}].plot2d(ax=ax, norm=LogNorm())

                mplhep.cms.label(ax=ax, **cms_label_kwargs)

                plt.annotate(TF_labels[iTF], (0.2, 0.82), xycoords="figure fraction", color="grey")
                if FULL_TAG: plt.annotate(full_tag, (8, 590), xycoords="axes points", color="black", fontsize=16)
                plt.tight_layout()

                fig.savefig(f"{outpath}/pdf/trg_vs_PF_{var}_{iTF}_SingleMu.pdf")
                fig.savefig(f"{outpath}/png/trg_vs_PF_{var}_{iTF}_SingleMu.png")
                plt.close()

    if "BX0" in full_tag or "BXa" in full_tag:
        plot_eff_2d(histograms)
        plot_eff_TFs(histograms)
        plot_eff_trigWPs(histograms)
        plot_probe_vs_tf_dR(histograms)  # TODO fix in process_events for BXn

    plot_probe_vs_tfmatch(histograms)
    plot_tag_and_probe(histograms)
    plot_tag_vs_probe(histograms)
    # plot_tag_vs_tf(histograms)
    plot_trig_probe_eta_vs_phi(histograms)

    print(f"plotting: done, took {time.time() - local_time}s")
