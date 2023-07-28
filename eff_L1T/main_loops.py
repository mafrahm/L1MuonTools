from config import dataset_Run3Mu_355872, dataset_Run3Mu_357479, dataset_Run3Mu_357735, dataset_Run2ZSkim, datasets_Run3Mu_357735_357479
from load_files import load_files
from init_histograms import init_histograms
from process_events import process_events
from plotting import plotting

#datasets = [dataset_Run3Mu_355872, dataset_Run2ZSkim]
datasets = [dataset_Run3Mu_355872, dataset_Run3Mu_357479, dataset_Run3Mu_357735, datasets_Run3Mu_357735_357479]

for dataset in datasets:
    infilename, files_key, RUN = load_files(dataset)
    histograms = init_histograms(RUN)

    #values_prb_pt = [0., 26.]
    #values_pt_sort = [True, False]
    #values_take_last = [True, False]
    #values_leading_tag = [True, False]

    #for prb_pt in values_prb_pt:
    #for leading_tag in values_leading_tag:

    #for pt_sort in values_pt_sort:
    #for take_last in values_take_last:
    #for leading_tag in values_leading_tag:

    outpath, full_tag = process_events(infilename, histograms, RUN)

    plotting(outpath, RUN, files_key, full_tag)

print("From main: all done!")
