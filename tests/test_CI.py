import os

#Check weights
def test_weights_file_exists():
    path = "Best-Model-Weights/weights-nano.zip"
    assert os.path.isfile(path), f"{path} does not exist"

#Check structure
def test_required_folders_exist():
    required = [
        "Best-Model-Weights",
        "train",
        "tests",
        "valid"
    ]
    for folder in required:
        assert os.path.isdir(folder), f"{folder} is missing"

#check requirments.txt
def test_requirements_exists():
    assert os.path.isfile("requirements.txt")