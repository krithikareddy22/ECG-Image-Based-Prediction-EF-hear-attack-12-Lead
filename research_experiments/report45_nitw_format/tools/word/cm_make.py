import sys
sys.path.insert(0, "/tmp/claude-0/r45docx")
from cm_plot import draw
OUT = sys.argv[1]
draw([[76, 11, 6, 2], [13, 13, 13, 2], [13, 13, 15, 2], [6, 4, 4, 6]], OUT + "/cm_50s_imgc3_locked_test",
     "Best in the 50s: 55.3 % four-class accuracy",
     "Image-only four-class classifier (IMG-C3) on the locked test: 199 unseen ECGs, scored once.\n"
     "Every patient gets a class: 110 of 199 correct, macro-F1 0.453, 95 % CI 0.485–0.621.")
draw([[65, 0, 0, 0], [10, 0, 1, 0], [8, 0, 8, 0], [5, 0, 1, 1]], OUT + "/cm_70s_triage_locked_test",
     "Best in the 70s: 74.7 % accuracy on answered patients",
     "Frozen image-only triage model on the locked test: answers 99 of 199 unseen ECGs (49.7 %).\n"
     "The rest are referred to echo. 74 of 99 answers correct (95 % CI 0.660–0.837); no Mild calls.",
     referred=[95 - 65, 41 - 11, 43 - 16, 20 - 7],
     note="Cell percentages and recall count answered patients only; the grey column shows the referred patients of each class.")
