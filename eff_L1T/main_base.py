from load_files import load_files
infilename, files_key, RUN = load_files()

from init_histograms import init_histograms
histograms = init_histograms(RUN)

from process_events import process_events
outpath, full_tag = process_events(infilename, histograms, RUN)

from plotting import plotting
plotting(outpath, RUN, files_key, full_tag)

print("From main: all done!")
