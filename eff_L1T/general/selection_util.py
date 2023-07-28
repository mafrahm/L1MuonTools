import awkward as ak
from typing import Sequence

def apply_masks(events, masks, obj, new_obj=None, reduce=True, nObj=1):
    """
    function to quickly apply masks on objects and reduce events based on the number of this object
    """
    if new_obj == None:
        new_obj = obj

    ## initialize field for new_obj
    events[new_obj] = events[obj]
    if isinstance(masks, ak.highlevel.Array):
        masks = {"mask": masks}  ## transform mask into dict
    elif isinstance(masks, Sequence):
        ## transform Sequence into dict
        masks = {f"mask_{i}": masks[i] for i in range(len(masks))}

    for k, mask in masks.items():
        if callable(mask):
            mask = mask(events[new_obj])  # `mask` is a function that returns a mask
        events[new_obj] = events[new_obj][mask]  # apply mask on the `new_obj` field
        if reduce:
            events = events[ak.num(events[new_obj]) >= nObj]
            print("Number of events after", k, ":", len(events))
    return events


def minDeltaR(obj1, obj2, var_expr=lambda A, B: A.deltaR(B)):

    ## check whether both object fields are jagged in outermost axis
    is_jagged = lambda arr: str(len(arr)) + " * var" in str(arr.type)
    if not ((is_jagged(obj1)) & (is_jagged(obj2))):
        raise NotImplementedError("In deltaR_cleaning: both objects {obj1} and {obj2} require a jagged structure in the outermost axis.")

    obj1, obj2 = ak.unzip(ak.cartesian({"obj1": obj1,"obj2": obj2}, nested=True))

    var = var_expr(obj1, obj2)

    print("var", var)
    print("min", ak.min(var, axis=-1))
    print("obj1", obj1.eta)
    print("obj2", obj2.eta)
    return ak.min(var, axis=-1)

## TODO better name: it's not only deltaR cleaning anymore but could be anything
def deltaR_cleaning(events, to_clean, clean_against, var_expr=lambda A, B: A.deltaR(B), var_req=lambda dR: dR < 0.4, assign_matches=True, reduce_to_clean=True, reduce_events=True):
    """
    function to quickly apply delta R matching between two jagged objects
    if reduce_to_clean==False: returns mask which can be applied on object `to_clean`
    if reduce_to_clean==True: returns events with reduced field to_clean and optionally (reduce_events==True) reduced events if no match is found
    NOTE: there might be issues if the structure of the objects is deeper than 2 axes
    """

    ## get the object arrays
    to_clean_arr = events[to_clean]
    clean_against_arr = events[clean_against]

    ## check whether both object fields are jagged in outermost axis
    is_jagged = lambda arr: str(len(arr)) + " * var" in str(arr.type)
    if not ((is_jagged(to_clean_arr)) & (is_jagged(clean_against_arr))):
        raise NotImplementedError("In deltaR_cleaning: both objects {to_clean} and {clean_against} require a jagged structure in the outermost axis.")

    ## reshape arrays into matrices
    to_clean_arr, clean_against_arr = ak.unzip(ak.cartesian({"to_clean": to_clean_arr,"clean_against": clean_against_arr}, nested=True))

    ## calculate discriminating variable
    var = var_expr(to_clean_arr, clean_against_arr)

    ## all `clean_aginst` objects are matched to the corresponding `to_clean`
    # index = clean_against_arr[var_req(var)].index
    matches = clean_against_arr[var_req(var)]

    matches["match_var"] = var[var_req(var)] ## assign the matching variable to the matched object
    mask = ak.num(matches, axis=2) > 0  ## axis=-1 should be the same for the considered cases, but does not always work for some reason

    ## assign the index of to_clean
    if assign_matches:
        events[to_clean, clean_against+"_matches"] = matches

    if reduce_to_clean:
        events = apply_masks(events, {f"matched {to_clean} to {clean_against}": mask}, to_clean, reduce=reduce_events)
        return events
    else:
        return mask
