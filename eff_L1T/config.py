from glob import glob
import time
import math
import numpy as np

## Configuration settings for different datasets

dataset_Run3Mu_eraC = {
    "RUN": 3,
    "ftype": "Mu",
    "run_str": "eraC",
    "fnames": glob("/eos/cms/store/user/eyigitba/l1t/dpg/run3/L1TNtuples/uGMTCrossCleaning/L1TNtuple_Run3_Muon_data_13p6TeV_eraC_uGMTCleaning_conservative_v1/*/*/*.root"),
}


MC_Run3DY = {
    "RUN": 3,
    "ftype": "DY",
    "run_str": "",
    "fnames": glob("/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut//DYToLL_M-50_TuneCP5_14TeV-pythia8/L1TNtuple_muonEff_Run3_baseline_v1/220417_091214/*.root"),
}

dataset_Run3Mu_357735 = {  ## 24 Files
    "RUN": 3,
    "ftype": "Mu",
    "run_str": 357735,
    "fnames": glob("/eos/cms/store/user/eyigitba/l1t/dpg/run3/L1TNtuples/august22/Muon/L1Ntuple_Muon_run2BDT_357735_IdealAlignment_20220826_092344/220826_142359/0000/*.root"),
}
dataset_Run3Mu_357479 = {  ## 38 Files
    "RUN": 3,
    "ftype": "Mu",
    "run_str": 357479,
    #"fnames": glob("/eos/cms/store/user/eyigitba/l1t/dpg/run3/L1TNtuples/august22/Muon/L1Ntuple_Muon_run3BDT_357479_IdealAlignment_20220825_115538/220825_165552/0000/*.root"),
    "fnames": glob("/eos/cms/store/user/eyigitba/l1t/dpg/run3/L1TNtuples/august22/Muon/L1Ntuple_Muon_run3BDT_357479_Run2Alignment_20220825_114411/220825_164425/0000/*.root"),
}
dataset_Run3Mu_355872 = {  ## 51 Files
    "RUN": 3,
    "ftype": "Mu",
    "run_str": "355872",
    "fnames": glob("/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut//SingleMuon/L1TNtuple_Run3_SingleMuon_data_13p6TeV_355872_v1/*/*//*.root"),
}
dataset_Run3Mu_355380 = {  ## 6 Files
    "RUN": 3,
    "ftype": "Mu",
    "run_str": "355380",
    "fnames": glob("/eos/cms/store/user/eyigitba/emtf/L1Ntuples/Run3/crabOut/SingleMuon/L1TNtuple_Run3_SingleMuon_data_13p6TeV_355380_v1/*/0000/*.root"),
}
dataset_Run3ZB = {
    "RUN": 3,
    "ftype": "ZB",
    "run_str": "355207_355208",
    "fnames": glob("/eos/cms/store/user/eyigitba/l1t/dpg/run3/L1TNtuples/jamboree13p6TeV/ZB/run355207_run355208/*.root"),
}
dataset_Run2ZSkim = {  ## 756 Files
    "RUN": 2,
    "ftype": "ZSkim",
    "run_str": "323413to324420",
    "fnames": glob(f"/eos/cms/store/user/arkadios/L1Ntpl/SingleMuon/SingleMuon_2018D_v2_runRange_323413to324420//181013_092226/*/*.root"),
}

## Configuration settings for file opening
#DATASET = datasets_Run3Mu_357735_357479
DATASET    = dataset_Run3Mu_eraC   ## choose the dataset of interest
MAX_FILE   = 9999                    ## Maximum number of input files
#BASE_PATH = "./data"                ## base path where to store everything
#BASE_PATH  = "/nfs/dust/cms/user/frahmmat/L1MuonTools/data"
BASE_PATH = "/eos/user/m/mfrahm/www/L1T/"
REDO_INPUT = False                  ## decide if the input file should be reprocessed
USE_EMUL   = False                  ## Use emulated L1T muons instead of unpacked

## Configuration settings for processing

REDO_HISTOGRAMS = True

REQ_BXi = "n"  ## Require L1T muon to be in BX 0. Options: int, "positive"/"p", "negative"/"n", "default"/"d"
             ## when using an option other than 0/"default", some probe muon requirements are added
#REQ_BX0  = True   ## Require L1T muon to be in BX 0
#REQ_BXnegative = True  ## Require L1T muon to be in BX -1 or -2 (overrides REQ_BX0 and changes some probe muon requirements)
#REQ_BXi = 1 ## Require L1T muon to be in BX i (overrides other REQ_BX's and changes some probe muon requirements)
REQ_uGMT = True    ## Require a final uGMT candidate, not just a TF muon
REQ_HLT  = False   ## Require tag muon to be matched to unprescaled HLT muon (removes all events in Run3 at the moment)
REQ_Z    = True   ## Require tag and probe muon to satisfy 81 < mass < 101 GeV

MAX_dR  = 0.4    ## Maximum dR for L1T-offline matching
TAG_ISO = 0.1    ## Maximum relative isolation for tag muon
TAG_PT  = 26.    ## Minimum offline pT for tag muon
PRB_PT  = 22.    ## Minimum offline pT for probe muon

LEADING_TAG   = False    ## only consider the leading tag   muon (-> dependent on how RecoMuon is sorted)
LEADING_PROBE = False   ## only consider the leading probe muon (-> dependent on how RecoMuon is sorted)
PT_SORT       = True    ## Decide if fields are pt-sorted (only relevant as long as only a sub-set of Tags/Probes is used)
TAKE_LAST     = False   ## If LEADING TAG or LEADING PROBE: use lowest-pt Tag and/or Probe Muon (TODO: I don't like the variable name)
SUFFIX        = ""      ## to append to outpath. If you hard-code some changes, use this to separate the output filenames

## Trigger settings
trig_WP = {  ## L1T Quality requirement for probe muon
    "SingleMu": [12, 13, 14, 15],
    "SingleMu7": [11, 12, 13, 14, 15],
    "DoubleMu": [8, 9, 10, 11, 12, 13, 14, 15],
    "MuOpen": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "TFMatch": [0],
}
trig_PT = {  ## Minimum L1T pT for probe muon
    "SingleMu": 22,
    "SingleMu7": 7,
    "DoubleMu": 8,
    "MuOpen": 3,
    "TFMatch": 0,
}
trig_TF = {  ## be careful to not loop over trig_TF since it includes both BMTf and KBMTf
    "uGMT": (0.00, 2.40),
    "BMTf": (0.00, 0.83),
    "KBMTf": (0.00, 0.83),
    "OMTf": (0.83, 1.24),
    "EMTf": (1.24, 2.40),
}

## Configuration settings (histograms)
OVERFLOW_BIN = True

BINS_DICT = {
    "pt": [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 45, 60, 75, 100, 140, 150],
    "eta": np.arange(-2.5, 2.51, 0.05),
    "phi": np.arange(-math.pi, math.pi+0.01, 2*math.pi / 72),
    #"eta": np.arange(-2.5, 2.51, 5 / 30),
    #"phi": np.arange(-math.pi, math.pi+0.01, 2*math.pi / 24),

}
LABELS = {
    "pt": "$p_{T}$ (Probe Reco $\mu$) [GeV]",
    "eta": "$\eta$ (Probe Reco $\mu$)",
    "phi": "$\phi$ (Probe Reco $\mu$)",
}

## Configuration settings (plotting)
FULL_TAG = True
LOGX_PT = False
