from general.loading_util import store_fields
import awkward as ak
import math

## Parameters that are changeable in main script
from config import DATASET, MAX_FILE, BASE_PATH, REDO_INPUT, USE_EMUL


def load_files(DATASET=DATASET, MAX_FILE=MAX_FILE, BASE_PATH=BASE_PATH, REDO_INPUT=REDO_INPUT, USE_EMUL=USE_EMUL):
    ## setup based on the chosen DATASET
    RUN = DATASET.get("RUN", "")
    files_key = f"Run{str(RUN)}{DATASET.get('ftype', '')}_{DATASET.get('run_str', '000000')}"
    if USE_EMUL: files_key += "_Emu"
    fnames = DATASET.get("fnames")
    N_files = len(fnames)
    print(files_key)
    print(f"Max. possible number of files from {files_key}: {N_files}")

    ## Trees and fields of interest (depend on the RUN variable)
    tree_names = {
        "Event": "l1EventTree/L1EventTree",
        "RecoMuon": "l1MuonRecoTree/Muon2RecoTree",
        "RecoMuonHlt": "l1MuonRecoTree/Muon2RecoTree",
        "L1Muon": "l1UpgradeTree/L1UpgradeTree/L1Upgrade",
        "OMTfMuon": "l1UpgradeTfMuonTree/L1UpgradeTfMuonTree/L1UpgradeOmtfMuon",
        "EMTfMuon": "l1UpgradeTfMuonTree/L1UpgradeTfMuonTree/L1UpgradeEmtfMuon",
    }
    tree_fields = {
        "Event": ["event"],
        "RecoMuon": ['pt', 'isMediumMuon', 'isTightMuon', 'iso', 'etaSt2', 'phiSt2'],
        "RecoMuonHlt": ["hlt_isomu", "hlt_isoDeltaR"],
        "L1Muon": ['muonIEt', 'muonIEta', 'muonBx'],
        "OMTfMuon": ['tfMuonHwPt', 'tfMuonHwEta', 'tfMuonHwQual', 'tfMuonGlobalPhi', "tfMuonBx"],
        "EMTfMuon": ['tfMuonHwPt', 'tfMuonHwEta', 'tfMuonHwQual', 'tfMuonGlobalPhi', "tfMuonBx"],
    }
    if RUN == 2:
        tree_names["BMTfMuon"] = "l1UpgradeTfMuonTree/L1UpgradeTfMuonTree/L1UpgradeBmtfMuon"
        tree_fields["BMTfMuon"] = ['tfMuonHwPt', 'tfMuonHwEta', 'tfMuonHwQual', 'tfMuonGlobalPhi', "tfMuonBx"]
    else:
        tree_names["KBMTfMuon"] = "l1UpgradeTfMuonTree/L1UpgradeTfMuonTree/L1UpgradeKBmtfMuon"
        tree_fields["KBMTfMuon"] = ['tfMuonHwPt', 'tfMuonHwEta', 'tfMuonHwQual', 'tfMuonGlobalPhi', "tfMuonBx"]

    if USE_EMUL:
        for k in tree_names.keys():
            tree_names["L1Muon"] = "l1UpgradeEmuTree/L1UpgradeTree/L1Upgrade"
            tree_names[k] = tree_names[k].replace("l1UpgradeTfMuonTree/l1UpgradeTfMuonTree/", "l1UpgradeTfMuonEmuTree/l1UpgradeTfMuonTree/")

    ## calling the function to produce the parquet file
    infilename = store_fields(fnames, files_key, tree_names, tree_fields, max_files=MAX_FILE, outdir=BASE_PATH, redo=REDO_INPUT, preprocess=preprocess_diMuon)

    return infilename, files_key, RUN


def preprocess_diMuon(events):
    ## merging of RecoMuonHlt into RecoMuon
    if "RecoMuonHlt" in events.fields:
        ## maximum of 20 RecoMuons
        events["RecoMuonHlt"] = events.RecoMuonHlt[:, :20]

        ## add RecoMuonHlt fields to RecoMuon
        for f in events.RecoMuonHlt.fields:
            events["RecoMuon", "hlt_"+f] = events.RecoMuonHlt[f]

    ## require at least two RecoMuon fulfilling general criteria
    mask = (events.RecoMuon.etaSt2 > -99) & (events.RecoMuon.phiSt2 > -99) & (events.RecoMuon.isMediumMuon) & (events.RecoMuon.pt > 1)
    events["RecoMuon"] = events.RecoMuon[mask]
    events = events[ak.num(events.RecoMuon) > 1]

    ## Transform RecoMuon and TFMuon fields into 4-vectors
    events["RecoMuon", "eta"] = events.RecoMuon.etaSt2  # using etaSt2 instead of eta
    events["RecoMuon", "phi"] = events.RecoMuon.phiSt2  # using phiSt2 instead of phi

    ## Combine all three iTFMuon fields into a single TfMuon field with additional field TF containing [(K)BMTf, OMTf, EMTf]
    TFs = [iTF for iTF in events.fields if "TfMuon" in iTF]
    TfMuons = []
    for iTFMuon in TFs:
        events[iTFMuon, "TF"] = iTFMuon
        TfMuons.append(events[iTFMuon])
    events["TfMuon"] = ak.concatenate(TfMuons, axis=1)

    ## remove iTf fields and RecoMuonHlt field
    keep_fields = ["Event", "RecoMuon", "TfMuon", "L1Muon"]
    events = ak.zip({f: events[f] for f in keep_fields}, depth_limit=1)

    ## transformation from hardware values to physical values
    events["TfMuon", "pt"] = (events["TfMuon"].tfMuonHwPt - 1) * 0.5
    events["TfMuon", "eta"] = events["TfMuon"].tfMuonHwEta * 0.010875
    phi = events["TfMuon"].tfMuonGlobalPhi * math.pi / 180
    phi = phi * 1.0 / 1.6  ## For some reason, it seems this is filled in a buggy way and needs to be rescaled - AWB 22.04.2019
    events["TfMuon", "phi"] = ak.where(phi > math.pi, phi - 2 * math.pi, phi)
    if ak.any(events["TfMuon"].phi > math.pi) or ak.any(events["TfMuon"].phi < -math.pi):
        raise ValueError("Values of phi should always be between -pi and +pi")

    ## add index field to TF Muons and RecoMuons (NOTE: not used yet, might not be needed at all)
    events["TfMuon", "index"] = ak.local_index(events["TfMuon"])
    events["RecoMuon", "index"] = ak.local_index(events["RecoMuon"])

    return events
