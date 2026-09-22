import argparse
import pathlib
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf
from keras import models
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.layers import Normalization, Resizing


DATASET_PATH = pathlib.Path("data")
MODEL_PATH = pathlib.Path("model.h5")
SAMPLE_RATE = 16000
SAMPLES_PER_CLIP = 16000


def get_label_names():
    """Return the class names in the same order used by the dataset loader."""
    return np.array(
        sorted(
            directory.name
            for directory in DATASET_PATH.iterdir()
            if directory.is_dir() and any(directory.glob("*.wav"))
        )
    )


def get_spectrogram(waveform):
    # Convert a waveform to the image-like input expected by the CNN.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    return tf.abs(spectrogram)[..., tf.newaxis]


def squeeze(audio, labels):
    return tf.squeeze(audio, axis=-1), labels


def build_and_train_model(label_names):
    train_data, validation_data = tf.keras.utils.audio_dataset_from_directory(
        DATASET_PATH,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=SAMPLES_PER_CLIP,
        subset="both",
    )
    train_data = train_data.map(squeeze, tf.data.AUTOTUNE)
    validation_data = validation_data.map(squeeze, tf.data.AUTOTUNE)

    def make_spectrogram_dataset(dataset):
        return dataset.map(
            lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_spectrograms = make_spectrogram_dataset(train_data).cache().shuffle(10000)
    train_spectrograms = train_spectrograms.prefetch(tf.data.AUTOTUNE)
    validation_spectrograms = make_spectrogram_dataset(validation_data).cache()
    validation_spectrograms = validation_spectrograms.prefetch(tf.data.AUTOTUNE)

    example_spectrograms, _ = next(iter(train_spectrograms.take(1)))
    input_shape = example_spectrograms.shape[1:]

    normalization = Normalization()
    normalization.adapt(train_spectrograms.map(lambda spec, label: spec))

    model = models.Sequential(
        [
            Input(shape=input_shape),
            Resizing(32, 32),
            normalization,
            Conv2D(32, 3, activation="relu"),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(len(label_names)),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_spectrograms,
        validation_data=validation_spectrograms,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return model


def load_or_train_model(label_names, retrain=False):
    if MODEL_PATH.exists() and not retrain:
        print(f"Loading model from {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    if retrain:
        print("Retraining requested...")
    else:
        print(f"{MODEL_PATH} was not found. Training a new model...")
    return build_and_train_model(label_names)


def prepare_waveform(audio):
    audio = tf.cast(audio, tf.float32)
    audio = audio[:SAMPLES_PER_CLIP]
    padding = SAMPLES_PER_CLIP - tf.shape(audio)[0]
    return tf.pad(audio, [[0, padding]])


def load_audio_file(file_path):
    audio_bytes = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(
        audio_bytes, desired_channels=1, desired_samples=SAMPLES_PER_CLIP
    )
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(
            f"'{file_path}' has a {int(sample_rate)} Hz sample rate; "
            f"expected {SAMPLE_RATE} Hz."
        )
    return prepare_waveform(tf.squeeze(waveform, axis=-1))


def record_live_audio(duration):
    try:
        import pyaudio
    except ImportError as error:
        raise RuntimeError(
            "Live recording requires PyAudio. Install it with: pip install pyaudio"
        ) from error

    audio = pyaudio.PyAudio()
    frames = []
    stream = None
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024,
        )
        print(f"Listening for {duration:g} seconds...")
        for _ in range(int(SAMPLE_RATE * duration / 1024)):
            frames.append(stream.read(1024, exception_on_overflow=False))
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()

    raw_audio = b"".join(frames)
    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    return prepare_waveform(tf.convert_to_tensor(samples))


def predict(model, waveform, label_names):
    spectrogram = get_spectrogram(waveform)[tf.newaxis, ...]
    probabilities = tf.nn.softmax(model(spectrogram)[0]).numpy()
    best_index = int(np.argmax(probabilities))
    print(f"Prediction: {label_names[best_index]}")
    print(f"Confidence: {probabilities[best_index] * 100:.2f}%")
    print("\nClass probabilities:")
    for label, probability in zip(label_names, probabilities):
        print(f"  {label}: {probability * 100:.2f}%")
    return label_names[best_index], probabilities[best_index]


class SpeechCommandApp:
    def __init__(self, root, label_names):
        self.root = root
        self.label_names = label_names
        self.model = None
        self.busy = False

        root.title("Speech Command Classifier")
        root.geometry("620x440")
        root.minsize(520, 360)

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill="both", expand=True)
        ttk.Label(
            main_frame,
            text="Speech Command Classifier",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            main_frame,
            text="Choose an audio file or record a command from the microphone.",
        ).pack(anchor="w", pady=(4, 18))

        controls = ttk.LabelFrame(main_frame, text="Audio input", padding=12)
        controls.pack(fill="x")
        self.file_button = ttk.Button(
            controls, text="Choose WAV file", command=self.choose_file
        )
        self.file_button.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.live_button = ttk.Button(
            controls, text="Record from microphone", command=self.record_live
        )
        self.live_button.grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Label(controls, text="Duration (seconds):").grid(
            row=0, column=2, padx=(16, 6)
        )
        self.duration = tk.StringVar(value="1")
        ttk.Entry(controls, textvariable=self.duration, width=7).grid(
            row=0, column=3, sticky="ew"
        )
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        actions = ttk.Frame(main_frame)
        actions.pack(fill="x", pady=14)
        self.retrain_button = ttk.Button(
            actions, text="Retrain model", command=self.retrain
        )
        self.retrain_button.pack(side="left")
        ttk.Button(actions, text="Clear result", command=self.clear_result).pack(
            side="left", padx=8
        )

        self.status = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status).pack(anchor="w", pady=(2, 8))
        self.result = tk.Text(main_frame, height=10, state="disabled", wrap="word")
        self.result.pack(fill="both", expand=True)

    def set_busy(self, busy, status):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.file_button.configure(state=state)
        self.live_button.configure(state=state)
        self.retrain_button.configure(state=state)
        self.status.set(status)

    def show_result(self, prediction, confidence):
        self.result.configure(state="normal")
        self.result.delete("1.0", tk.END)
        self.result.insert(
            tk.END,
            f"Prediction: {prediction}\nConfidence: {confidence * 100:.2f}%\n\n"
            f"Available commands: {', '.join(self.label_names)}",
        )
        self.result.configure(state="disabled")

    def run_background(self, action, status):
        if self.busy:
            return
        self.set_busy(True, status)

        def worker():
            try:
                prediction, confidence = action()
                self.root.after(0, lambda: self.show_result(prediction, confidence))
                self.root.after(0, lambda: self.set_busy(False, "Done"))
            except Exception as error:
                self.root.after(0, lambda: messagebox.showerror("Error", str(error)))
                self.root.after(0, lambda: self.set_busy(False, "An error occurred"))

        threading.Thread(target=worker, daemon=True).start()

    def get_model(self, retrain=False):
        if self.model is None or retrain:
            self.model = load_or_train_model(self.label_names, retrain)
        return self.model

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not file_path:
            return

        def action():
            return predict(self.get_model(), load_audio_file(file_path), self.label_names)

        self.run_background(action, "Loading file and predicting...")

    def record_live(self):
        try:
            duration = float(self.duration.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid duration", "Enter a positive number of seconds.")
            return

        def action():
            return predict(
                self.get_model(), record_live_audio(duration), self.label_names
            )

        self.run_background(action, "Listening...")

    def retrain(self):
        def action():
            self.model = load_or_train_model(self.label_names, retrain=True)
            return "Model ready", 1.0

        self.run_background(action, "Training model...")

    def clear_result(self):
        self.result.configure(state="normal")
        self.result.delete("1.0", tk.END)
        self.result.configure(state="disabled")
        self.status.set("Ready")


def run_gui():
    label_names = get_label_names()
    if not len(label_names):
        raise RuntimeError(f"No WAV command folders found in {DATASET_PATH}")
    root = tk.Tk()
    SpeechCommandApp(root, label_names)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Speech command classifier")
    parser.add_argument(
        "--cli", action="store_true", help="Use command-line mode instead of the GUI"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", type=pathlib.Path, help="Path to a 16 kHz WAV file")
    source.add_argument(
        "--live", action="store_true", help="Record audio from the microphone"
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Live recording duration in seconds"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Ignore model.h5 and train again"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.cli and not args.file and not args.live and not args.retrain:
        run_gui()
        return

    label_names = get_label_names()
    if not len(label_names):
        raise RuntimeError(f"No WAV command folders found in {DATASET_PATH}")

    model = load_or_train_model(label_names, args.retrain)
    if args.file:
        waveform = load_audio_file(str(args.file))
    elif args.live:
        waveform = record_live_audio(args.duration)
    else:
        choice = input("Enter 'f' for an audio file or 'l' for live microphone: ").strip().lower()
        if choice == "f":
            waveform = load_audio_file(input("Audio file path: ").strip())
        elif choice == "l":
            waveform = record_live_audio(args.duration)
        else:
            raise ValueError("Choose 'f' or 'l'.")
    predict(model, waveform, label_names)


if __name__ == "__main__":
    main()
