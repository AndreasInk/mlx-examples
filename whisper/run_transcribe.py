from whisper import transcribe
import os
import torch
from tqdm import tqdm
import pandas as pd

class ParkinsonSpeechDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Parkinson speech data.
    Expects a 1:1 mapping between audio files (.wav or .m4a) and transcript files (.txt).
    If a transcript file is missing, it creates one using part of the file path.
    """
    def __init__(self, data_dir, device="cpu"):
        self.audio_files = []
        self.text_files = []

        # Recursively search for audio and transcript files.
        for root, _, files in os.walk(data_dir):
            audio_files = sorted([os.path.join(root, f) for f in files if f.endswith('.wav') or f.endswith('.m4a')])
            text_files = sorted([os.path.join(root, f) for f in files if f.endswith('.txt')])

            # Ensure a 1:1 mapping between audio and transcript files.
            if len(audio_files) != len(text_files):
                for file in audio_files:
                    # If a transcript is missing, create one using part of the file path.
                    new_txt_file = file.replace(".m4a", ".txt").replace(".wav", ".txt")
                    if not os.path.exists(new_txt_file):
                        with open(new_txt_file, "w") as f:
                            # Use a substring of the path as a dummy transcript.
                            f.write(new_txt_file.split(".txt")[0].split("/SI/")[-1])
                        text_files.append(new_txt_file)

            self.audio_files.extend(audio_files)
            self.text_files.extend(text_files)

        assert len(self.audio_files) == len(self.text_files), "Mismatch between audio and transcript files"
        self.device = device

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        text_path = self.text_files[idx]

        # Load and clean the reference transcript.
        with open(text_path, 'r') as f:
            transcript = f.read().strip()

        return audio_path, transcript


if __name__ == "__main__":
    """
    python transcribe.py \
      --audio ./some/path/to/some-audio-file.m4a.or.wav \
      --model ./some_mlx_model/tiny_fp16
    """
    dataset = ParkinsonSpeechDataset("/Users/andreas/Desktop/Whisper+PD/original-speech-dataset/SI/")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    results = []
    actuals = []
    for idx, (audio_paths, transcripts) in enumerate(tqdm(loader, desc="Evaluating")):
        result = transcribe(
            audio_paths[0],
            path_or_hf_repo="/Users/andreas/Documents/mlx-examples/whisper/lora/lora_fused_model_whisper_with_PD",
            initial_prompt=(
                "This is a conversation between a patient and a Parkinson’s management application. "
                "The discussion includes topics such as symptom tracking, medication schedules, and exercise routines. "
                "Keywords include: tremor, rigidity, bradykinesia, dyskinesia, levodopa, carbidopa, sinemet, dopamine, entacapone, amantadine, "
                "deep brain stimulation, freezing of gait, balance, physical therapy, speech therapy, occupational therapy, daily logs, and medication adherence."
            ),
            language="en"
        )
        results.append(result["text"])
        actuals.append(transcripts[0])
    df = pd.DataFrame()
    df["predicted"] = results
    df["actual"] = actuals
    df.to_csv("./whisper_finetuned.csv")

