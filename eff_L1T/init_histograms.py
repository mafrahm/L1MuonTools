import hist
import numpy as np

## Parameters that are fixed by config
from config import BINS_DICT, LABELS, trig_WP


def init_histograms(RUN):
    TFs = ["KBMTf", "OMTf", "EMTf"]
    if RUN == 2:
        TFs = ["BMTf", "OMTf", "EMTf"]

    ## Initialize histograms
    histograms = {}

    histograms["eta_vs_phi"] = ( ## 2d Probe Muon (denominator)
        hist.Hist.new
        .StrCat(TFs+["uGMT"], name="TF")
        .Var(BINS_DICT["eta"], name="probe_eta", label=LABELS["eta"])
        .Var(BINS_DICT["phi"], name="probe_phi", label=LABELS["phi"])
        .Double()
    )
    histograms["trg_eta_vs_phi"] = ( ## 2d Triggered Probe Muon (numerator)
        hist.Hist.new
        .StrCat(TFs+["uGMT"], name="TF")
        .StrCat(list(trig_WP.keys()), name="trig_WP")
        .Var(BINS_DICT["eta"], name="probe_eta", label=LABELS["eta"])
        .Var(BINS_DICT["phi"], name="probe_phi", label=LABELS["phi"])
        .Double()
    )

    for var, bins in BINS_DICT.items():
        histograms[var] = (  ## Probe Muon (denominator)
            hist.Hist.new
            .StrCat(TFs+["uGMT"], name="TF")
            .Var(bins, name="probe_"+var, label=LABELS[var])
            .Double()
        )
        histograms["tag_"+var] = (  ## Tag Muon
            hist.Hist.new
            .StrCat(TFs+["uGMT"], name="TF")
            .Var(bins, name="tag_"+var, label=LABELS[var].replace("Probe", "Tag"))
            .Double()
        )
        tag_v_probe_bins = bins
        if(var=="pt"): tag_v_probe_bins = np.arange(0, 150, 2)
        histograms["tag_vs_probe_"+var] = (  ## Tag vs Probe Muon
            hist.Hist.new
            .StrCat(TFs+["uGMT"], name="TF")
            .Var(tag_v_probe_bins, name="probe_"+var, label=LABELS[var])
            .Var(tag_v_probe_bins, name="tag_"+var, label=LABELS[var].replace("Probe", "Tag"))
            .Double()
        )
        histograms["probe_vs_tfmatch_"+var] = (  ## Tag vs TfMatch Muon
            hist.Hist.new
            .StrCat(TFs+["uGMT"], name="TF")
            .Var(tag_v_probe_bins, name="probe_"+var, label=LABELS[var])
            .Var(tag_v_probe_bins, name="tfmatch_"+var, label=LABELS[var].replace("Probe Reco", "TF match"))
            .Double()
        )
        histograms["probe_"+var+"_vs_tf_dR"] = ( ## Tag vs dR(Probe, triggered Tf without dR req)
            hist.Hist.new
            .StrCat(TFs+["uGMT"], name="TF")
            .Var(tag_v_probe_bins, name="probe_"+var, label=LABELS[var])
            .Var(np.arange(0, 5.01, 0.05), name="tftrig_dR", label="min dR(Probe, TfTrig)")
            .Double()
        )
        histograms[f"trg_{var}"] = (  ## Triggered Probe Muon
            hist.Hist.new
            .StrCat(TFs+["uGMT"], name="TF")
            .StrCat(list(trig_WP.keys()), name="trig_WP")
            .Var(bins, name="probe_"+var, label=LABELS[var])
            .Double()
        )
        if var != "pt":
            histograms[f"trg_{var}_vs_PF"] = (  ## Reco Probe vs TF Probe Muon
                hist.Hist.new
                .StrCat(TFs+["uGMT"], name="TF")
                .StrCat(list(trig_WP.keys()), name="trig_WP")
                .Var(bins, name="probe_"+var, label=LABELS[var])
                .Var(bins, name="PF_"+var, label=LABELS[var].replace("Reco", "TF"))
                .Double()
            )

    histograms["nTags_vs_nProbes"] = (
        hist.Hist.new
        .Int(0, 5, name="nTags", label="nTags")
        .Int(0, 5, name="nProbes", label="nProbes")
        .Double()
    )
    histograms["tag_probe_mass"] = (
        hist.Hist.new
        .Var(np.arange(60, 120.01, 2), name="tag_probe_mass", label=r"M($\mu_{Tag}+\mu_{Probe}$)")
        .Double()
    )

    return histograms
