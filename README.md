# AI Speech Command Classifier

A deep learning application for recognizing short voice commands from audio files or a live microphone. It uses TensorFlow and a convolutional neural network (CNN) to classify speech commands, with a graphical interface built with Tkinter.

## Features

- Classifies eight commands: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`
- Loads `model.h5` automatically when it exists
- Trains and saves a new model when no trained model is available
- Supports WAV files and live microphone recording
- Provides both a graphical interface and command-line mode
- Displays the predicted command and confidence score
- Supports forced model retraining

## How It Works

1. The waveform is converted into a spectrogram using STFT.
2. The spectrogram is resized and normalized.
3. A CNN analyzes the spectrogram and predicts the command.
4. The application displays the prediction and confidence.

## Project Structure

```text
.
├── index.py          # Main application
├── model.h5          # Trained model, if available
├── README.md
└── data/
	├── down/
	├── go/
	├── left/
	├── no/
	├── right/
	├── stop/
	├── up/
	└── yes/
```

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Keras
- PyAudio for microphone input

Install the required packages:

```bash
pip install tensorflow numpy pyaudio
```

## Run the Graphical Interface

The GUI is the default mode:

```bash
python index.py
```

The window lets you choose a WAV file, record from the microphone, set the recording duration, retrain the model, and view the prediction.

## Command-Line Usage

Predict a command from a WAV file:

```bash
python index.py --cli --file data/stop/example.wav
```

Record and classify audio from the microphone:

```bash
python index.py --cli --live --duration 1
```

Retrain the model:

```bash
python index.py --cli --retrain
```

View all options:

```bash
python index.py --help
```

## Audio Format

Input files must be WAV files sampled at **16,000 Hz**. Audio longer than one second is trimmed, while shorter audio is padded with zeros.

## Model Behavior

- If `model.h5` exists, it is loaded and used immediately.
- If it does not exist, the model is trained using the folders inside `data/` and then saved.
- The `--retrain` option forces a new training run.

## Dataset

The audio samples are based on the [Google Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). This repository contains a local excerpt organized by command label. For the complete dataset, see the [official TensorFlow dataset source](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz).

## License and Data Notice

Please review the original Speech Commands dataset documentation and license terms before redistributing the audio data.