# Supporting code for "Automatic time in bed detection from hip-worn accelerometers for large epidemiological studies: The Tromsø Study"

This repository contains the code to train the model reported in 

Weitz, M., Syed, S., Hopstock, L. A., Morseth, B., André Henriksen, & Horsch, 
A. (2024). Automatic time in bed detection from hip-worn accelerometers for 
large epidemiological studies: The Tromsø Study. Under review at *PLOS ONE*.

For data access, contact the Tromsø Study Data and Publication Committee 
(tromsous@uit.no), The Tromsø Study, Department of Community Medicine, Faculty 
of Health Sciences, UiT The Arctic University of Norway.


## How to use

1. Obtain the raw acceleration data files from the Tromsø Study as described 
above. Note, that this data is not freely available and that the Tromsø 
Study charges a fee for using it.

2. Install all relevant dependencies using 
[pypoetry](https://python-poetry.org/).

3. Run the `create_1min_data.py` to obtain the 1min aggregated data, that
was used in the paper.

4. Run the `train.py` script to train the models.

5. Feel free to adapt this script for the Tromsø Study or any other data
you have. If you think it contributed to your scientific work, we are 
delighted if you cite the above-mentioned paper.


## Contact information

In case, you have any questions, do not hesitate to contact me 
(marc.weitz@uit.no).