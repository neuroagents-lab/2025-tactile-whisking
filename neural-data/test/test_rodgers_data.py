import numpy as np
from dataset.rodgers_data import load_rodgers_data, concatenate_sessions, concat_other_animals

def test_concatenate_sessions():
    """
    Test the concatenate_sessions function.
    """
    session1 = np.full((100, 2, 30), 1)
    session2 = np.full((150, 2, 35), 2)
    concatenated = concatenate_sessions([session1, session2])
    
    assert concatenated.shape == (150, 2, 65), "Concatenated shape is incorrect"
    assert np.all(concatenated[:100, :, :30] == session1), "Session 1 data mismatch"
    assert np.all(concatenated[:150, :, 30:65] == session2), "Session 2 data mismatch"


def test_concat_other_animals():
    """
    Test the concat_other_animals function.
    """
    animal_id = "animal_1"
    data_per_animal = {
        "animal_1": np.full((100, 2, 30), 1),
        "animal_2": np.full((150, 2, 35), 2),
        "animal_3": np.full((120, 2, 40), 3)
    }
    
    concatenated = concat_other_animals(animal_id, data_per_animal)
    
    assert isinstance(concatenated, np.ndarray), "Concatenated data should be a numpy array"
    assert concatenated.shape == (150, 2, 75), "Concatenated shape is incorrect"


def test_load_rodgers_data():
    """
    Test the load_rodgers_data function.
    """
    rodgers_data = load_rodgers_data(path="./data/rodgers6_data.npz")
    assert isinstance(rodgers_data, dict), "Data should be a dictionary"
    assert len(rodgers_data) > 0, "Data should not be empty"
    for animal, sessions in rodgers_data.items():
        assert isinstance(animal, str), "Animal ID should be a string"
        assert isinstance(sessions, np.ndarray), "Sessions should be a numpy array"
        assert len(sessions.shape) == 3, "Sessions should be 3 dimensional (trials, stimuli, units)"
