Python script for detection of medical stitches in images. 
Authors: Jiří Belinger, Ondřej Sládek

Testing data (images) are stored in folder "images".
For detection of stitches in different data, store images chosen for detection to folder "images" before script execution.
Results (text/visual) will be stored to subfolder "results" created in folder "src" after script execution.
Requirements concerning external python libraries are defined in file "requirements.txt".

Script for detection - "run.py" file.
Script arguments: [1] - name of output csv file (e.g. "output.csv") with text results - stored to "results" folder.
                  [2] - execution of visual mode ("-v" for YES, nothing for NO).
                  [(2),3,4,5,...] - names of images chosen for detection (must be available in "images" folder).

Script was tested for python versions 3.10.12 and 3.11.9.
