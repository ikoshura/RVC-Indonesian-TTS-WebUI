import asyncio
import datetime
import logging
import os
import time
import traceback
import subprocess
import torch
import gradio as gr
import librosa
from g2p_id import G2P
import ast
import re
from fairseq import checkpoint_utils
import soundfile as sf
import numpy as np
np.int = int


from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

# Initialize G2P module for Indonesian
g2p = G2P()

# Coqui TTS functions
def get_speaker_idxs():
    """Get speaker mappings from Coqui TTS model"""
    try:
        result = subprocess.run(
            "tts --model_path checkpoint.pth --config_path config.json --list_speaker_idxs",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
        start_index = output.find("{")
        end_index = output.rfind("}")
        if start_index != -1 and end_index != -1:
            dict_str = output[start_index:end_index+1]
            speaker_dict = ast.literal_eval(dict_str)
            mapping = {}
            counter = 1
            for key in speaker_dict.keys():
                if re.match(r"^(JV|SU)-\d+$", key):
                    if key.startswith("JV"):
                        mapping[key] = f"Pembicara {counter} (Jawa)"
                    else:  # key pasti diawali SU
                        mapping[key] = f"Pembicara {counter} (Sunda)"
                    counter += 1
                else:
                    mapping[key] = key
            return mapping
        return {}
    except Exception as e:
        print("Error getting speaker indexes:", e)
        return {}

speaker_mapping = get_speaker_idxs()
friendly_names = list(speaker_mapping.values())
reverse_mapping = {v: k for k, v in speaker_mapping.items()}

def generate_tts(text, speaker):
    """Generate TTS using Coqui Indonesian model"""
    original_key = reverse_mapping.get(speaker, "SU-0")
    
    # Convert text to phonemes
    phoneme_text = g2p(text)
    print(f"Phoneme conversion: {text} -> {phoneme_text}")
    
    # Generate TTS
    command = (
        f"tts --text \"{phoneme_text}\" "
        f"--model_path checkpoint.pth "
        f"--config_path config.json "
        f"--speaker_idx {original_key} "
        f"--out_path coqui_output.wav"
    )
    subprocess.run(command, shell=True, check=True)
    
    # Convert to 16kHz mono for RVC processing
    audio, sr = librosa.load("coqui_output.wav", sr=16000, mono=True)
    sf.write("coqui_output.wav", audio, sr)
    return "coqui_output.wav"

# RVC Configuration
model_root = "weights"
models = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
models.sort()

def model_data(model_name):
    """Load RVC model data"""
    pth_path = os.path.join(model_root, model_name, [f for f in os.listdir(os.path.join(model_root, model_name)) if f.endswith(".pth")][0])
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    if version == "v1":
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half) if if_f0 == 1 else SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half) if if_f0 == 1 else SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()
    
    vc = VC(tgt_sr, config)
    index_files = [f for f in os.listdir(os.path.join(model_root, model_name)) if f.endswith(".index")]
    index_file = index_files[0] if index_files else ""
    
    return tgt_sr, net_g, vc, version, index_file, if_f0

# Load models
print("Loading hubert model...")
hubert_model = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")[0][0]
hubert_model = hubert_model.to(config.device)
hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
hubert_model.eval()

print("Loading rmvpe model...")
rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)

def rvc_convert(
    model_name,
    tts_text,
    speaker,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25
):
    """Main conversion pipeline"""
    try:
        if limitation and len(tts_text) > 280:
            return "Error: Text too long for this environment", None, None

        # Generate TTS using Coqui
        tts_file = generate_tts(tts_text, speaker)
        
        # Load RVC model
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        
        # Process audio
        audio, sr = librosa.load(tts_file, sr=16000, mono=True)
        if limitation and len(audio)/sr > 20:
            return "Error: Audio too long", tts_file, None

        # Voice conversion
        f0_up_key = int(f0_up_key)
        times = [0, 0, 0]
        
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            tts_file,
            times,
            f0_up_key,
            f0_method,
            index_file,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        
        info = f"Success. Processing times: npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        return info, tts_file, (tgt_sr, audio_opt)
    
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", None, None

# Gradio Interface
initial_md = """
# RVC Indonesian TTS WebUI

Indonesian Text ➡[Coqui TTS]➡ Audio ➡[RVC]➡ Converted Voice
"""

with gr.Blocks() as app:
    gr.Markdown(initial_md)
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(label="RVC Model", choices=models, value=models[0])
            f0_key_up = gr.Number(label="Pitch Shift", value=0)
            f0_method = gr.Radio(
                label="Pitch Extraction Method",
                choices=["pm", "rmvpe"],
                value="rmvpe"
            )
            index_rate = gr.Slider(0, 1, value=1, label="Index Rate")
            protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect")
        with gr.Column():
            tts_text = gr.Textbox(label="Input Text", placeholder="Masukkan teks Indonesia...")
            speaker = gr.Dropdown(label="Speaker", choices=friendly_names, value=friendly_names[0])
            convert_btn = gr.Button("Convert", variant="primary")
        with gr.Column():
            info_output = gr.Textbox(label="Status")
            orig_audio = gr.Audio(label="Original Audio")
            rvc_audio = gr.Audio(label="Converted Audio")
    
    convert_btn.click(
        rvc_convert,
        [model_name, tts_text, speaker, f0_key_up, f0_method, index_rate, protect],
        [info_output, orig_audio, rvc_audio]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)