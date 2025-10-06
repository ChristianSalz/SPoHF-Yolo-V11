# SPoHF-YOLOv8 Project

This is the **public repository** for the SPoHF project (https://spohf.com/).  
All ongoing development, experiments, and updates can be followed here.  

The project focuses on building and training YOLOv8 models for insect detection and classification.  
Please note that dataset folders are intentionally empty — users need to download or provide their own data to run the experiments.  

> **Important note:**  
> We also have a YOLOv12 expirimental branch (https://github.com/ChristianSalz/YoloV12-Insect-Detection) to try out, but please be aware that these project is not officially supported by the ultralyrtics community so performance and stability may vary.


# Installation Instructions (Mac)

1. Install Homebrew (if not already installed)

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Verify installation:

```bash
brew --version
```

2. Install Python 3.13 via Homebrew

```bash
brew install python@3.13
```

Make sure your shell uses the correct version:

```bash
brew link python@3.13 --force --overwrite
python3 --version
```

3. Navigate to the Project Folder

```bash
cd /path/to/YoloV12-Insect-Detection
```

4. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

5. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

6. Download the data from The SPoHF-Roboflow-Project and add the Train/Valid/Test folder to your project
   https://app.roboflow.com/spohf-insect-counting/spohf-kur4x-dokg9/models - go to 'versions' (left menu), select the version, then press 'download dataset'. Copy over the content in the project directory.

7. Use the trainTheModel.py file to train a yolo v8 model - Currenltly MPS (Apple Metal) is supported, but stability may vary - as fallback scenario train on the CPU

8. Run the SPoHF-predict.py to test your model PS: update the path to your trained model - example (unseen) data is provide in the `Manual-Test-Data` folder

# Installation Instructions (Windows)

1. Go and buy a mac
2. Follow steps above

# Installation Instructions (Linux)

You guys are pros, you dont need installation instructions :)

# Concept
