import argparse
import torch
import librosa
import numpy as np
from pathlib import Path
from sonics import HFAudioClassifier

# Model configurations
MODEL_MAP = {
    "alpha-5s": "awsaf49/sonics-spectttra-alpha-5s",
    "beta-5s": "awsaf49/sonics-spectttra-beta-5s",
    "gamma-5s": "awsaf49/sonics-spectttra-gamma-5s",
    "alpha-120s": "awsaf49/sonics-spectttra-alpha-120s",
    "beta-120s": "awsaf49/sonics-spectttra-beta-120s",
    "gamma-120s": "awsaf49/sonics-spectttra-gamma-120s",
}


def load_model(model_name, device):
    """Load SpecTTTra model"""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_MAP.keys())}")

    model_id = MODEL_MAP[model_name]
    print(f"Loading model: {model_id}")
    model = HFAudioClassifier.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model


def process_audio(audio_path, model, device, chunk_strategy="middle"):
    """Process audio file and return prediction"""
    max_time = model.config.audio.max_time

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = int(max_time * sr)

    # Handle chunk extraction based on strategy
    if chunk_strategy == "middle":
        total_chunks = len(audio) // chunk_samples
        chunk_idx = total_chunks // 2
        start = chunk_idx * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]
    elif chunk_strategy == "first":
        chunk = audio[:chunk_samples]
    elif chunk_strategy == "average":
        # Average predictions across all chunks
        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)

        preds = []
        with torch.no_grad():
            for chunk in chunks:
                chunk_tensor = torch.from_numpy(chunk).float().to(device)
                pred = model(chunk_tensor.unsqueeze(0))
                prob = torch.sigmoid(pred).cpu().numpy()[0]
                preds.append(prob)

        prob = np.mean(preds)
        return {"real": float(1 - prob), "fake": float(prob)}
    else:
        raise ValueError(
            f"Invalid chunk strategy: {chunk_strategy}. "
            "Choose from: middle, first, average"
        )

    # Pad if necessary
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

    # Get prediction
    with torch.no_grad():
        chunk_tensor = torch.from_numpy(chunk).float().to(device)
        pred = model(chunk_tensor.unsqueeze(0))
        prob = torch.sigmoid(pred).cpu().numpy()[0]

    return {"real": float(1 - prob), "fake": float(prob)}


def main():
    parser = argparse.ArgumentParser(
        description="SpecTTTra Inference Script for Audio Classification"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input audio file or directory",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gamma-5s",
        choices=list(MODEL_MAP.keys()),
        help="Model to use for inference (default: gamma-5s)",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto)",
    )

    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="middle",
        choices=["middle", "first", "average"],
        help="Chunk extraction strategy (default: middle)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "-o", "--output", type=str, help="Output file to save results (CSV)"
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device)

    # Process input
    input_path = Path(args.input)
    results = []

    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        # Support common audio formats
        audio_files = []
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a"]:
            audio_files.extend(input_path.glob(ext))
        audio_files.sort()
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not audio_files:
        print("No audio files found!")
        return

    print(f"\nProcessing {len(audio_files)} file(s)...\n")

    for audio_file in audio_files:
        try:
            prediction = process_audio(str(audio_file), model, device, args.strategy)

            real_prob = prediction["real"]
            fake_prob = prediction["fake"]
            label = "REAL" if real_prob > args.threshold else "FAKE"
            confidence = max(real_prob, fake_prob)

            result = {
                "file": audio_file.name,
                "label": label,
                "confidence": confidence,
                "real_prob": real_prob,
                "fake_prob": fake_prob,
            }

            results.append(result)

            if args.verbose:
                print(f"File: {audio_file.name}")
                print(f"  Label: {label}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Real: {real_prob:.4f} | Fake: {fake_prob:.4f}\n")
            else:
                print(f"{audio_file.name}: {label} (confidence: {confidence:.4f})")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")

    # Save results if output specified
    if args.output:
        import csv

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "label", "confidence", "real_prob", "fake_prob"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
