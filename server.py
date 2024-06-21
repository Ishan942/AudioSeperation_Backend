from flask import Flask, request, jsonify
from flask_cors import CORS # CORS for handling Cross-Origin Resource Sharing
import pickle 
import torch
import torchaudio
import base64
import io
# Printing the versions of torch and torchaudio
print(torch.__version__)
print(torchaudio.__version__)

import matplotlib.pyplot as plt
from IPython.display import Audio
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from torchaudio.transforms import Fade
import os


# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app,resources={r"/*":{"origins":"*"}})

################################################################################
bundle = HDEMUCS_HIGH_MUSDB_PLUS

# Getting the model from the bundle
model = bundle.get_model()

# Loading the trained model's state dictionary from a file
model.load_state_dict(torch.load('hdemucs_model.pth', map_location='cpu'))

# Transfer the model to CPU
device = torch.device("cpu")
model.to(device)

# Get the sample rate from the bundle
sample_rate = bundle.sample_rate

# Printing the sample rate
print(f"Sample rate: {sample_rate}")
################################################################################


def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    """
    Apply a model to a given mixture, using fading and adding segments together to process the model segment by segment.

    Args:
        model (torch.nn.Module): The model to apply to the mixture.
        mix (torch.Tensor): The input mixture tensor with shape (batch, channels, length).
        segment (float): Segment length in seconds.
        overlap (float): Overlap factor between segments (0 to 1).
        device (torch.device, str, or None): Device on which to execute the computation.
            If None, the device of the input mix tensor is assumed.

    Returns:
        torch.Tensor: The separated sources tensor with shape (batch, sources, channels, length).
    """
    # Set the device for computation
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    # Get the sample rate from the model bundle
    sample_rate = bundle.sample_rate
    # Calculate chunk length and overlap frames
    chunk_len = int(sample_rate * segment * (1 + overlap))
    overlap_frames = overlap * sample_rate

    # Initialize fade transform
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    # Initialize output tensor
    batch, channels, length = mix.shape
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    # Process the mixture segment by segment
    start = 0
    end = chunk_len
    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out

        # Update start and end indices for next segment
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len

        # Adjust fade-out length for the last segment
        if end >= length:
            fade.fade_out_len = 0

    return final


# Define a route for handling HTTP GET requests to the root URL
@app.route('/', methods=['GET'])
def get_data():
    data = {
        "message":"API is Running"
    }
    return jsonify(data)
  
# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['audio_file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.to(device)
    mixture = waveform

    # Print waveform and sample rate information
    print("Waveform: ", waveform)
    print("Sample Rate: ", sample_rate)

    # Set parameters for separating tracks
    segment: int = 10
    overlap = 0.1

    print("Separating track")

    # Perform normalization on the waveform
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    # Separate sources using the `separate_sources` function
    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
    )[0]

    # Denormalize the separated sources
    sources = sources * ref.std() + ref.mean()

    # Convert the separated sources to a list and create an audio dictionary
    sources_list = model.sources
    sources = list(sources)
    audios = dict(zip(sources_list, sources))

    output_data = {}
    for source_name, source_audio in audios.items():
        buffer = io.BytesIO()
        torchaudio.save(buffer, source_audio.cpu(), sample_rate, format="wav")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        output_data[source_name] = audio_base64

    return jsonify(output_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)