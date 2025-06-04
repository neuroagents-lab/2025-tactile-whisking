# Mouse Data

Since WHISKiT provides a rat model, we need to first obtain the mouse whisker parameters.

The MATLAB scripts are from the original WHISKiT repo. 
As a note, we are not interested in `generate_whisk` since we are not doing active whisking.

The mouse whisker data is sourced from [Bresee et al. 2023](https://github.com/SeNSE-lab/BreseeEtAl_JEB2023_dataFiles/tree/v1)

1. Open `compute_parameters` folder in MATLAB
2. Run `compute_parameters('average', '../../data/', 1)`
3. Copy over to `whiskitphysics/code/data/whisker_param_average_mouse`