import os, subprocess, shutil
#import torch
import gradio as gr
from slicer2 import Slicer
import librosa
import soundfile as sf

output_dir = "output"
pretrained_model_path = "pre_trained_model/vec768l12"

def preprocess_slice(input_dir):
    slicer_params = {
        "threshold": -40,
        "min_length": 5000,
        "min_interval": 300,
        "hop_size": 10,
        "max_sil_kept": 500,
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    original_min_interval = slicer_params["min_interval"]
    def process_audio(filename):
        audio, sr = librosa.load(os.path.join(input_dir, filename), sr=None, mono=False)
        slicer = Slicer(sr=sr, **slicer_params)
        chunks = slicer.slice(audio)
        files_to_delete = []
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            output_filename = f"{os.path.splitext(filename)[0]}_{i}"
            output_filename = "".join(c for c in output_filename if c.isascii() or c == "_") + ".wav"
            output_filepath = os.path.join(output_dir, output_filename)
            sf.write(output_filepath, chunk, sr)
            # re-slice
            while True:
                new_audio, sr = librosa.load(output_filepath, sr=None, mono=False)
                if librosa.get_duration(y=new_audio, sr=sr) <= 15:
                    break
                slicer_params["min_interval"] = slicer_params["min_interval"] // 2 
                if slicer_params["min_interval"] >= slicer_params["hop_size"]: 
                    new_chunks = Slicer(sr=sr, **slicer_params).slice(new_audio)
                    for j, new_chunk in enumerate(new_chunks):
                        if len(new_chunk.shape) > 1:
                            new_chunk = new_chunk.T
                        new_output_filename = f"{os.path.splitext(output_filename)[0]}_{j}.wav"
                        sf.write(os.path.join(output_dir, new_output_filename), new_chunk, sr)
                    files_to_delete.append(output_filepath)
                else:
                    break
            slicer_params["min_interval"] = original_min_interval
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            process_audio(filename)

    for filename in os.listdir(output_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(output_dir, filename)
            audio, sr = librosa.load(filepath, sr=None, mono=False)
            if librosa.get_duration(y=audio, sr=sr) < 2:
                os.remove(filepath)
    slicer_params["min_interval"] = original_min_interval
    return output_dir

def train():
    for file in os.listdir("logs/44k"):
        if file != "diffusion":
            shutil.rmtree(file)
    d_0_path = os.path.join(pretrained_model_path, "D_0.pth")
    g_0_path = os.path.join(pretrained_model_path, "G_0.pth")
    shutil.copy(d_0_path, os.path.join("logs/44k", "D_0.pth"))
    shutil.copy(g_0_path, os.path.join("logs/44k", "G_0.pth"))
    train_cmd = r"..\so-vits-svc\workenv\python.exe train.py -c configs/config.json -m 44k"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", train_cmd])
    return "Training successfully started."

def pipeline(input_dir):
    output = preprocess_slice(input_dir)
    if not os.path.exists("dataset_raw/speaker"):   
        os.makedirs("dataset_raw/speaker")
    for wav in os.listdir(output):
        if wav.endswith(".wav"):
            shutil.move(os.path.join(output, wav), "dataset_raw/speaker")
    preprocess_commands = [
        r"..\so-vits-svc\workenv\python.exe resample.py",
        r"..\so-vits-svc\workenv\python.exe preprocess_flist_config.py --speech_encoder vec768l12",
        r"..\so-vits-svc\workenv\python.exe preprocess_hubert_f0.py --f0_predictor dio"
        ]
    accumulated_output = ""
    dataset = os.listdir("dataset/44k")
    if len(dataset) != 0:
        for dir in dataset:
            dataset_dir = os.path.join("dataset/44k", dir)
            if os.path.isdir(dataset_dir):
                shutil.rmtree(dataset_dir)
                accumulated_output += f"Deleting previous dataset: {dir}\n"
    for command in preprocess_commands:
        try:
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
            accumulated_output += f"Command: {command}\n"
            yield accumulated_output
            progress_line = None
            for line in result.stdout:
                if r"it/s" in line or r"s/it" in line:
                    progress_line = line
                else:
                    accumulated_output += line
                if progress_line is None:
                    yield accumulated_output
                else:
                    yield accumulated_output + progress_line
            result.communicate()
        except subprocess.CalledProcessError as e:
            result = e.output
            accumulated_output += f"Error: {result}\s"
            yield accumulated_output
    if progress_line is not None:
        accumulated_output += progress_line
    accumulated_output += '-' * 50 + '\n'
    yield accumulated_output, None
    msg = train()
    accumulated_output += msg
    yield accumulated_output

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("Demo"):
            raw_audio_path = gr.Textbox(label="Raw audio path(a folder that includes all raw audios)")
            submit_btn = gr.Button("Submit", variant = "primary")
            debug_msg = gr.Textbox(label = "Debug Message")

    submit_btn.click(pipeline, [raw_audio_path], [debug_msg])
    app.queue(concurrency_count=1022, max_size=2044).launch()