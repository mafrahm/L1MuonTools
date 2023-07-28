import os
import time
from typing import Sequence

import pickle
import math
import numpy as np
import awkward as ak
import hist
import particle
import vector
vector.register_awkward()

from general.selection_util import apply_masks, deltaR_cleaning, minDeltaR

## Parameters that are changeable in main script
from config import (
    REDO_HISTOGRAMS, REQ_BXi, REQ_uGMT, REQ_HLT, REQ_Z, MAX_dR, TAG_ISO, TAG_PT, PRB_PT,
    LEADING_TAG, LEADING_PROBE, PT_SORT, TAKE_LAST, SUFFIX,
)

## Parameters that are fixed by config
from config import trig_WP, trig_PT, trig_TF, OVERFLOW_BIN, BINS_DICT

def process_events(
    infilename, histograms, RUN,
    REDO_HISTOGRAMS=REDO_HISTOGRAMS, REQ_BXi=REQ_BXi,
    REQ_uGMT=REQ_uGMT, REQ_HLT=REQ_HLT, REQ_Z=REQ_Z,
    MAX_dR=MAX_dR, TAG_ISO=TAG_ISO, TAG_PT=TAG_PT, PRB_PT=PRB_PT, LEADING_TAG=LEADING_TAG,
    LEADING_PROBE=LEADING_PROBE, PT_SORT=PT_SORT, TAKE_LAST=TAKE_LAST, SUFFIX=SUFFIX,
):
    local_time = time.time()
    TFs = ["KBMTf", "OMTf", "EMTf"]
    if RUN == 2:
        TFs = ["BMTf", "OMTf", "EMTf"]

    # disable BX0 when BXnegative is True, disable BXn when BXi is not 0

    def test_int(s):
        try:
            int(s)
            return True
        except:
            return False

    if test_int(REQ_BXi):
        BX_str = str(REQ_BXi)
        BX_mask = lambda d: d.tfMuonBx == int(REQ_BXi)
    else:
        # options: "a" (all BX's), "d" (default=BX0), "p" (positive, BX>0), "n" (negative, BX<0)
        BX_str = REQ_BXi[0].lower()
        if BX_str == "d":
            BX_str = "0"
            BX_mask = lambda d: d.tfMuonBx == 0
        elif BX_str == "a": BX_mask = lambda d: Ellipsis
        elif BX_str == "p": BX_mask = lambda d: d.tfMuonBx > 0
        elif BX_str == "n": BX_mask = lambda d: d.tfMuonBx < 0
        else: raise Exception("Invalid option for paramter REQ_BXi.")

    ## prepare unique outpath name
    data_path = "/".join(infilename.split("/")[:-1])
    foldername =  (
        f"tag{int(TAG_PT)}_prb{int(PRB_PT)}" + (f"_BX{BX_str}") +
        ("_uGMT" if REQ_uGMT else "") + ("_HLT" if REQ_HLT else "") + ("_Z" if REQ_Z else "") +
        ("_1Tag" if LEADING_TAG else "") + ("_1Prb" if LEADING_PROBE else "")
    )
    if LEADING_TAG or LEADING_PROBE:
        foldername += ("_ptSort" if PT_SORT else "_noSort") + ("_takeLast" if TAKE_LAST else "")
    foldername += ("_" + SUFFIX if SUFFIX else "")

    outpath = f"{data_path}/{foldername}"
    outfilename = f"{outpath}/histograms.pickle"

    full_tag = foldername + f"_iso{str(TAG_ISO).replace('.', 'p')}_dr{str(MAX_dR).replace('.', 'p')}"

    ## create outpath folder
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    ## check if output already exist
    if os.path.exists(outfilename) and not REDO_HISTOGRAMS:
        print(".... In process_events: file already exist: \n", outfilename)
        return outpath, full_tag

    ## define requirements for different types of muons

    ## requirements for both the Tag and the Probe Muon (already implemented in preprocessing)
    recomuon_masks = {
        "eta_phi": lambda d: (d.etaSt2 > -99) & (d.phiSt2 > -99),
        "recoIsMed": lambda d: d.isMediumMuon,
        #"recoIsTight": lambda d: d.isTightMuon,
        #"eta_test": lambda d: abs(d.etaSt2) < 2.4,
    }
    ## requirements on the tag muon (dR matching not included)
    tag_masks = {
        "tag_iso": lambda d: d.iso < TAG_ISO,
        "tag_pt": lambda d: d.pt > TAG_PT,
    }
    if REQ_HLT:
        tag_masks["hlt_isomu"] = lambda d: d.hlt_isomu != 0           ## in Run3, no event survives isomu cut
        tag_masks["hlt_isoDeltaR"] = lambda d: d.hlt_isoDeltaR < 0.1  ## in Run3, no event survives isoDeltaR cut

    ## requirements for the tf tag muon
    tf_tag_masks = {
        "tf_qual": lambda d: d.tfMuonHwQual >= trig_WP["SingleMu"][0],
        "tf_pt": lambda d: d.pt > TAG_PT - 4.01,  # .01 to convert to float or what?
    }
    ## if REQ_BX0: tf_tag_masks["bx0"] = lambda d: d.tfMuonBx == 0  # in the old script, this is implemented later (after filling denominator histograms. Why?)

    ## requirements on the probe muon (dR cleaning not included)
    probe_masks = {
        "probe_pt": lambda d: d.pt > PRB_PT,
    }
    if BX_str not in ["0", "a"]:
        probe_masks["probe_tight"] = lambda d: d.isTightMuon
        # probe_masks["max_probe_pt"] = lambda d: d.pt < 22

    ## Read-in events
    events = ak.from_parquet(infilename)
    print("Number of events:", len(events))

    if PT_SORT:
        ## pt-sort All Muon fields
        events["RecoMuon"] = events["RecoMuon"][ak.argsort(events.RecoMuon.pt, axis=-1, ascending=False)]
        events["L1Muon"] = events["L1Muon"][ak.argsort(events.L1Muon.muonIEt, axis=-1, ascending=False)]
        events["TfMuon"] = events["TfMuon"][ak.argsort(events.TfMuon.tfMuonHwPt, axis=-1, ascending=False)]

    ## enable 4-vector behavior
    events["RecoMuon"] = ak.with_name(events.RecoMuon, "Momentum4D")
    events["TfMuon"] = ak.with_name(events["TfMuon"], "Momentum4D")

    ## assign mass field to RecoMuon and TfMuon
    events["RecoMuon", "mass"] = particle.Particle.from_pdgid(13).mass / 1000.0
    events["TfMuon", "mass"] = particle.Particle.from_pdgid(13).mass / 1000.0

    ## apply requirements on all reco muons and require at least two reco muons
    ## (already done in preprocessing, should not reduce events)
    events = apply_masks(events, recomuon_masks, "RecoMuon", reduce=True, nObj=2)

    ## generate TagMuon field, apply simple requirements
    events = apply_masks(events, tag_masks, "RecoMuon", new_obj="TagMuon", reduce=True)

    ## generate TF Tag field, apply simple requirements
    events = apply_masks(events, tf_tag_masks, "TfMuon", "TfTags", reduce=False)

    ## require at least 1 Reco TagMuon with dR match to a Tf TagMuon
    events = deltaR_cleaning(events, "TagMuon", "TfTags", var_req=lambda dR: dR < MAX_dR)

    print("Number of events with >1 Reco Tag Muons:", len(events[ak.num(events.TagMuon)>1]))
    if LEADING_TAG:
        if TAKE_LAST:
            print("Only consider the inverse-leading Reco TagMuon....")
            events["TagMuon"] = ak.from_regular(events["TagMuon"][:, [-1]])
        else:
            print("Only consider the leading Reco TagMuon....")
            events["TagMuon"] = ak.from_regular(events["TagMuon"][:, [0]])

    ## generate ProbeMuon field, apply simple requirements
    events = apply_masks(events, probe_masks, "RecoMuon", new_obj="ProbeMuon", reduce=True)

    ## Require at least 2*max_dr between Tag and Probe Reco Muons
    events = deltaR_cleaning(events, "ProbeMuon", "TagMuon", var_req=lambda dR: dR > 2 * MAX_dR)

    if REQ_Z:
        ## require an invariant mass M(Probe, Tag) close to the Z mass peak
        events = deltaR_cleaning(events, "ProbeMuon", "TagMuon", var_expr=lambda A, B: (A+B).mass, var_req=lambda mass: (mass > 81) & (mass < 101))

    print(f"Number of events with >1 Probe Muon:", len(events.ProbeMuon[ak.num(events.ProbeMuon)>1]))
    if LEADING_PROBE:
        if TAKE_LAST:
            print("Only consider the inverse-leading Reco Probe....")
            events["ProbeMuon"] = ak.from_regular(events["ProbeMuon"][:, [-1]])
        else:
            print("Only consider the leading Reco Probe....")
            events["ProbeMuon"] = ak.from_regular(events["ProbeMuon"][:, [0]])

    ## before starting to fill histograms, restrict pt values to last bin of pt histograms
    if OVERFLOW_BIN:
        max_pt = BINS_DICT["pt"][-1] - 0.01
        print("restrict Reco pt values to", max_pt, "(overflow bins)")
        events["ProbeMuon", "pt"] = ak.where(events.ProbeMuon.pt > max_pt, max_pt, events.ProbeMuon.pt)
        events["TagMuon", "pt"] = ak.where(events.TagMuon.pt > max_pt, max_pt, events.TagMuon.pt)

    iTF_mask = lambda d, iTF: (abs(d.eta) > trig_TF[iTF][0]) & (abs(d.eta) < trig_TF[iTF][1])


    ## require BXnegative for TF Probe
    if BX_str not in ["0", "a"]: events["TfMuon"] = events["TfMuon"][BX_mask(events.TfMuon)]
    ## reset histograms to not fill them over and over again
    for h in histograms.values():
        h.reset()

    print("====== Start filling histograms")
    ## Calculate the total number of Reco Tag and Probe Muons
    nTags = ak.num(events.TagMuon)
    nProbes = ak.num(events.ProbeMuon)
    histograms["nTags_vs_nProbes"].fill(nTags, nProbes)

    ## Calculate M(mu, mu) for each combination of Tag and Probe Muons
    mProbe, mTag = ak.broadcast_arrays(events.ProbeMuon[{"pt", "eta", "phi", "mass"}], events.ProbeMuon.TagMuon_matches[{"pt", "eta", "phi", "mass"}])
    mProbe = ak.with_name(mProbe, "Momentum4D")
    mTag = ak.with_name(mTag, "Momentum4D")
    histograms["tag_probe_mass"].fill(ak.ravel((mProbe+mTag).mass))

    print("====== Start filling denominator histograms")
    for var in BINS_DICT.keys():
        for iTF in TFs+["uGMT"]:
            mask = iTF_mask(events.ProbeMuon, iTF)
            if iTF=="uGMT": mask = ak.ones_like(mask)  ## NOTE: this means, probes with eta>2.4 are also filled in denominator but not in numerator.... is that fair?
            probe_var = ak.flatten(events.ProbeMuon[mask][var])
            histograms[var].fill(TF=iTF, **{"probe_"+var: probe_var})

            tag_mask = iTF_mask(events.TagMuon, iTF)
            if iTF=="uGMT": tag_mask = ak.ones_like(tag_mask)
            tag_var = ak.flatten(events.TagMuon[tag_mask][var])
            histograms["tag_"+var].fill(TF=iTF, **{"tag_"+var: tag_var})

            ## NOTE: for tag vs probe plots, just use leading tag for each probe
            tag_var_2d = ak.flatten(events.ProbeMuon[mask]).TagMuon_matches[:, 0][var]
            histograms["tag_vs_probe_"+var].fill(TF=iTF, **{"probe_"+var: probe_var, "tag_"+var: tag_var_2d})

    # 2d denominator distribution
    for iTF in TFs+["uGMT"]:
        mask = iTF_mask(events.ProbeMuon, iTF)
        if iTF=="uGMT": mask = ak.ones_like(mask)
        fill_kwargs = {
            "TF": iTF,
            "probe_eta": ak.flatten(events.ProbeMuon[mask].eta),
            "probe_phi": ak.flatten(events.ProbeMuon[mask].phi),
        }
        histograms["eta_vs_phi"].fill(**fill_kwargs)

    print("Number of events with valid Probe:", len(events))
    print("Number of valid Probes:", ak.sum(ak.num(events.ProbeMuon)))
    #for iTF in TFs+["uGMT"]: print(f"Number of events with valid Probe in {iTF}:", ak.sum(iTF_mask(events.ProbeMuon, iTF)))

    print("======= Start matching TF probe to Reco Probe Muon and filling numerator histograms")

    ## require BX0 for TF Probe (original options)
    if BX_str in ["0", "a"]: events["TfMuon"] = events["TfMuon"][BX_mask(events.TfMuon)]

    ## require probes to be inside the central eta region of 2.4 (NOTE: remove?)
    #events["ProbeMuon"] = events["ProbeMuon"][abs(events.ProbeMuon.eta < 2.4)]
    events = apply_masks(events, {"|eta|<2.4": lambda d: abs(d.eta) < 2.4}, "ProbeMuon")

    ## Match Tf Muon to uGMT Muon
    if REQ_uGMT:
        Tf, uGMT = ak.unzip(ak.cartesian({"TF": events.TfMuon,"uGMT": events.L1Muon}, nested=True))
        uGMT_masks = {
            "bx": Tf.tfMuonBx == uGMT.muonBx,
            "eta": Tf.tfMuonHwEta == uGMT.muonIEta,
            "pt": Tf.tfMuonHwPt == uGMT.muonIEt,
        }
        uGMT_mask = ak.sum((uGMT_masks["bx"] & uGMT_masks["eta"] & uGMT_masks["pt"]), axis=-1)>0

        # apply uGMT mask and remove events without a matched TfMuon
        events["TfMuon"] = events["TfMuon"][uGMT_mask]
        events = events[ak.num(events.TfMuon)>0]

    #TODO: fix
    if BX_str in ["0", "a"]:
        trig_SingleMu_mask = lambda d: (d.tfMuonHwQual >= trig_WP["SingleMu"][0]) & (d.pt > trig_PT["SingleMu"] - 0.01)
        events["TfTrig"] = events["TfMuon"][trig_SingleMu_mask(events.TfMuon)]
        events["ProbeMuon", "minDeltaR_tftrig"] = minDeltaR(events.ProbeMuon, events.TfTrig, var_expr=lambda A, B: A.deltaR(B))
        for iTF in TFs+["uGMT"]:
            mask = iTF_mask(events.ProbeMuon, iTF)
            prb = events.ProbeMuon[mask]
            prb = prb[ak.num(prb)>0]
            prb = prb[~ak.is_none(prb.minDeltaR_tftrig)]
            print(prb.pt)
            print(prb.minDeltaR_tftrig)
            for var in BINS_DICT.keys():
                fill_kwargs={"TF": iTF, "probe_"+var: ak.flatten(prb[var]), "tftrig_dR": ak.flatten(prb.minDeltaR_tftrig)}
                histograms["probe_"+var+"_vs_tf_dR"].fill(**fill_kwargs)

    ## Require dR matching between Tag and at least one TfMuon
    events = deltaR_cleaning(events, "ProbeMuon", "TfMuon", var_req=lambda dR: dR < MAX_dR)

    ## Filling of probe vs tf match histograms
    for iTF in TFs+["uGMT"]:
        mask = iTF_mask(events.ProbeMuon, iTF)
        prb = events.ProbeMuon[mask]
        prb = prb[ak.num(prb)>0]
        for var in BINS_DICT.keys():
            ## just take the main TfMuon match for each Probe
            fill_kwargs={"TF": iTF, "probe_"+var: ak.flatten(prb[var]), "tfmatch_"+var: ak.flatten(prb.TfMuon_matches[:, :, 0][var])}
            histograms["probe_vs_tfmatch_"+var].fill(**fill_kwargs)

    ## Loop over trigger working-points for efficiency measurements
    for k, WP in trig_WP.items():
        print(f"{k}: quality >= {WP[0]}, p_T > {trig_PT[k]}")

        ## determine for each ProbeMuon which matched TfMuon triggered
        trig_WP_mask = lambda d: (d.tfMuonHwQual >= WP[0]) & (d.pt > trig_PT[k] - 0.01)  # 0.01 to convert to float or what?
        events["ProbeMuon", "TfMuon_triggered"] = events["ProbeMuon", "TfMuon_matches"][trig_WP_mask(events.ProbeMuon.TfMuon_matches)]

        ## for testing both trigger requirements individually
        trig_pt_mask = lambda d: d.pt > trig_PT[k] - 0.01
        trig_qual_mask = lambda d: d.tfMuonHwQual >= WP[0]
        print(f"Number of Probes with TfMuon fulfilling pt   trigger requirement from {k}:", ak.sum(ak.num(events["ProbeMuon", "TfMuon_matches"][trig_pt_mask(events.ProbeMuon.TfMuon_matches)], axis=-1)>0))
        print(f"Number of Probes with TfMuon fulfilling qual trigger requirement from {k}:", ak.sum(ak.num(events["ProbeMuon", "TfMuon_matches"][trig_qual_mask(events.ProbeMuon.TfMuon_matches)], axis=-1)>0))

        ## read out all Probe Muons with a triggered TF Match
        iProbes = events.ProbeMuon[ak.num(events.ProbeMuon.TfMuon_triggered, axis=-1)>0]

        print(f"Number of events triggered from {k}:", ak.sum(ak.num(iProbes)))

        ## Filling of histograms with triggered objects
        for var in BINS_DICT.keys():
            for iTF in TFs+["uGMT"]:
                mask = iTF_mask(iProbes, iTF)
                fill_kwargs = {"TF": iTF, "trig_WP": k, "probe_"+var: ak.flatten(iProbes[mask][var])}
                histograms[f"trg_{var}"].fill(**fill_kwargs)

                ## eta/phi of reco muon vs matched TF muon (leading)
                if var != "pt":
                    fill_kwargs["PF_"+var] = ak.flatten(iProbes[mask]).TfMuon_triggered[:, 0][var]
                    histograms[f"trg_{var}_vs_PF"].fill(**fill_kwargs)

        ## Filling of 2d histograms with triggered objects
        for iTF in TFs+["uGMT"]:
            mask = iTF_mask(iProbes, iTF)
            fill_kwargs = {"TF": iTF, "trig_WP": k, "probe_eta": ak.flatten(iProbes[mask].eta), "probe_phi": ak.flatten(iProbes[mask].phi)}
            histograms[f"trg_eta_vs_phi"].fill(**fill_kwargs)


    ## Calculate and print-out inclusive efficiencies
    if BX_str in ["0", "a"]:
        from hist.intervals import clopper_pearson_interval
        for k in trig_WP.keys():
            print("WP:", k)
            for iTF in TFs+["uGMT"]:
                N_num = histograms["trg_phi"][{"trig_WP": k, "TF": iTF, "probe_phi": sum}]
                N_denom = histograms["phi"][{"TF": iTF, "probe_phi": sum}]
                eff = 100 * N_num / N_denom
                unc_simple = eff / math.sqrt(N_num)
                unc_cpi = clopper_pearson_interval(N_num, N_denom)
                print(f"{iTF}, efficiency: {eff:.2f} +- {unc_simple:.2f}")
                #print(f"{iTF}, efficiency: {eff:.2f} +{unc_cpi[0]:.2f} -{unc_cpi[0]:.2f}")

    ## save histograms as pickle file
    with open(f"{outpath}/histograms.pickle", "wb") as handle:
        pickle.dump(histograms, handle)

    print(f"process_events: done, took {time.time() - local_time}s")
    return outpath, full_tag
