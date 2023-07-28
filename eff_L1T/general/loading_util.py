import os
import time
from typing import Sequence

import uproot
import awkward as ak


def get_arrays(fnames,
               treename = "l1MuonRecoTree/Muon2RecoTree",
               filter_name = "*"
              ):
    in_files = {fname:treename for fname in fnames}
    arrays = uproot.concatenate(in_files, filter_name = filter_name, how="zip")
    return arrays


def combine_fields(fnames, tree_names, tree_fields):
    events = {}
    for key, treename in tree_names.items():
        print(f"{key} fields:")
        arr = get_arrays(fnames, treename=treename, filter_name=tree_fields[key])
        for field in arr.fields:
            if len(arr[field].fields) > 0: print(f"{[field]} with sub-fields {arr[field].fields}")
            else: print(field)

        if len(arr.fields)==1:
            print(f"remove unnecessary depth due to {arr.fields[0]}")
            arr = arr[arr.fields[0]]
        events[key] = arr
    events = ak.zip(events, depth_limit=1)
    return events


## TODO implement chunked reading?
def store_fields(fnames, files_key, tree_names, tree_fields, max_files=2, outdir="./data", redo=True, preprocess=None):
    local_time = time.time()
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print(f"files from key {files_key}")
    fname = f"{files_key}_{len(fnames[:max_files])}Files"
    if len(fnames[:max_files]) == 0:  ## when no files are found, at least try to find existing output
           fname = f"{files_key}_{max_files}Files"
    fname += ("_" + preprocess.__name__ if preprocess else "")

    if not os.path.isdir(f"{outdir}/{fname}"):
        os.mkdir(f"{outdir}/{fname}")

    outfilename = f"{outdir}/{fname}/{fname}.parquet"
    print(outfilename)
    if os.path.exists(outfilename) and not redo:
        print(f".... file already exist: {outfilename}")
    else:
        events = combine_fields(fnames[:max_files], tree_names, tree_fields)
        print("Main fields:", events.fields)
        print("Number of events:", len(events))

        if preprocess:
            if not callable(preprocess): raise ValueError(f"In store_fields: preprocess parameter expects a function")
            events = preprocess(events)
            print("Main fields after preprocessing:", events.fields)
            print("Number of events after preprocessing:", len(events))

        ak.to_parquet(events, outfilename)
        print(f".... file created: {outfilename}")

    print(f"store_fields: done, took {time.time() - local_time}s")
    return outfilename
