import warnings
import pynwb

# suppress warnings
warnings.filterwarnings("ignore", message="Ignoring cached namespace 'hdmf-common' version")
warnings.filterwarnings("ignore", message="Ignoring cached namespace 'core' version")
warnings.filterwarnings("ignore", message="Ignoring cached namespace 'hdmf-experimental' version")

def get_nwbfile(filepath):
    nwb_io = pynwb.NWBHDF5IO(filepath, "r")
    return nwb_io.read() 
    
