# RVC-Indonesian-TTS-WebUI

This is a Gradio-based text-to-speech WebUI for Indonesian TTS using a two-stage pipeline:
1. **Coqui TTS (Indonesian version):** Converts Indonesian text into speech via a phoneme-based model.
2. **RVC Conversion:** Processes the Coqui TTS output using RVC models to generate converted voice outputs.

This project builds upon the ideas from [RVC Text-to-Speech WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and integrates [g2p-id](https://github.com/Wikidepia/g2p-id) for grapheme-to-phoneme conversion tailored to Indonesian.

![image](https://github.com/user-attachments/assets/9d79c173-f5e4-4cab-a004-066008d19424)

> **Note:** This project is tested for Python 3.10 on Windows 11. Python 3.11 or later might not be supported at the moment.

---

## Install

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ikoshura/RVC-Indonesian-TTS-WebUI.git
   cd RVC-Indonesian-TTS-WebUI
   ```

2. **Download Required Models** 
    
    #### **Voice Conversion Model**  
    Download the required files from the following links:  
    - [`hubert_base.pt`](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)  
    - [`rmvpe.pt`](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt)  
    
    #### **Indonesian-TTS Models**  
    You can find Indonesian-TTS models in the [Releases](https://github.com/Wikidepia/indonesian-tts/releases/) section of the [indonesian-tts](https://github.com/Wikidepia/indonesian-tts) repository.  
    
    Make sure to place the following files in the same directory as `app.py` or update the paths in the code accordingly:  
    - Model checkpoint: `checkpoint.pth`  
    - Speaker embeddings: `speakers.pth`  
    - Configuration file: `config.json`  
    - Voice Conversion models: `hubert_base.pt`, `rmvpe.pt`

4. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

5. **Install PyTorch (Optional for GPU Users)**

   If you wish to utilize an NVIDIA GPU, install the compatible PyTorch version. For example (for CUDA 11.8 on Windows):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For CPU-only usage, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

6. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```
7. **Project Directory Structure**

    Ensure your directory is set up as follows:
    
    ```
    rvc-tts-webui/
    ├── assets/                  # Asset files (if applicable)
    ├── lib/                     # Library files
    ├── venv/                    # Virtual environment
    ├── weights/                 # Model weights directory
    ├── .gitignore               # Git ignore file
    ├── app.py                   # Main application script
    ├── checkpoint.pth           # Model checkpoint
    ├── config.json              # Model configuration
    ├── config.py                # Configuration script
    ├── hubert_base.pt           # Hubert model file
    ├── LICENSE                  # License file
    ├── README.md                # Project documentation
    ├── requirements.txt         # Dependencies file
    ├── rmvpe.pt                 # RMVPE model file
    ├── rmvpe.py                 # RMVPE processing script
    ├── speakers.pth             # Speaker embeddings
    └── vc_infer_pipeline.py     # Voice conversion inference pipeline
    ```

---

## Locate RVC Models

Place your RVC models in the `weights/` directory using the following structure:

```bash
weights
├── model1
│   ├── my_model1.pth
│   └── my_index_file_for_model1.index
└── model2
    ├── my_model2.pth
    └── my_index_file_for_model2.index
```

Each model directory should contain exactly one `.pth` file and optionally one `.index` file. The directory names will be used as the model names in the UI.

> **Tip:** Avoid using non-ASCII characters in the model paths (e.g., `weights/モデル1`) to prevent FAISS errors.

---

## Launch

Activate your virtual environment and run the application:

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

python app.py
```
or Run the application by opening the `run.bat` file.

The Gradio interface will launch in your default browser.

---

## Update

To update the project, pull the latest changes and reinstall the dependencies:

```bash
git pull
venv\Scripts\activate
pip install -r requirements.txt --upgrade
```

---

## Troubleshooting

If you encounter an error such as:

```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

It is likely that `fairseq` or another dependency requires Microsoft C++ Build Tools. Download and install them from:
[Download Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

## Project Overview

This project integrates several components:

- **Coqui TTS for Indonesian:**  
  The code leverages Coqui TTS to synthesize speech from text after converting Indonesian text to phonemes using [g2p-id](https://github.com/Wikidepia/g2p-id).

- **RVC Voice Conversion:**  
  After generating the initial TTS output, the audio is processed with RVC models for voice conversion. The pipeline loads models from the `weights/` directory and uses:
  - A pretrained Hubert model (loaded via fairseq).
  - The RMVPE model for pitch extraction.
  - Custom modules (`VC`, and synthesizer models) for voice conversion.

- **Gradio WebUI:**  
  The web interface allows users to input Indonesian text, select a speaker (with readable names generated from the Coqui TTS model), adjust pitch shift and other parameters, and view both the original and converted audio outputs.

The main code is located in `app.py` and related helper modules. Below is a snippet that shows key functions:

## Requirements

The **requirements.txt** file includes:

```text
torch
gradio
librosa
g2p-id
git+https://github.com/Tps-F/fairseq.git@main
soundfile
numpy
git+https://github.com/Wikidepia/g2p-id
```

> **Note:** If you run into version conflicts, please check the documentation for each package. It is recommended to use versions compatible with Python 3.10.

---

## Credits

- **Base Project:** [RVC Text-to-Speech WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- **TTS Engine:** [Coqui TTS](https://github.com/coqui-ai/TTS)
- **Indonesian Grapheme-to-Phoneme Conversion:** [g2p-id](https://github.com/Wikidepia/g2p-id)
- **indonesian-tts:** [indonesian-tts](https://github.com/Wikidepia/indonesian-tts)

---

Happy experimenting with Indonesian TTS and voice conversion!
