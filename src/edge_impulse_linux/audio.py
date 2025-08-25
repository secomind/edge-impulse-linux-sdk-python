"""Modified version of edge_impulse_linux.audio to make it work with Pyton 3.10.

The original version only works with Python 3.9.
The issue is in PyAudio dependency which may not be fully  compatible with Python 3.10,
so it has been necessary to modify the code and use a different approach for audio
capture and buffering: instead of relying heavily on PyAudio's callback mechanism,
it has been use a more explicit threading approach.
"""

import queue
import threading
from typing import Dict, List, Tuple

import numpy as np
import pyaudio

from edge_impulse_linux.runner import ImpulseRunner

CHUNK_SIZE = 1024
OVERLAP = 0.25


class Microphone:
    def __init__(
        self,
        rate: int,
        chunk_size: int,
        device_id: int | None = None,
        channels: int | None = 1,
    ):
        """Initialize the Microphone.

        Args:
            rate (int): Sampling rate.
            chunk_size (int): Size of audio chunks in frames.
            device_id (int, optional): Audio input device ID. If None, prompts the user.
            channels (int, optional): Number of input audio channels. Defaults to 1.
        """
        self.buff = queue.Queue()
        self.chunk_size = chunk_size
        self.rate = rate
        self.closed = True
        self.channels = channels
        self.interface = pyaudio.PyAudio()
        self.device_id = device_id
        self.zero_counter = 0
        self.thread = None
        self.stream = None

        while self.device_id is None or not self.check_device_compatibility(
            self.device_id
        ):
            input_devices = self.list_available_devices()
            input_device_id = int(
                input("Type the id of the audio device you want to use: \n")
            )
            for device in input_devices:
                if device[0] == input_device_id:
                    if self.check_device_compatibility(input_device_id):
                        self.device_id = input_device_id
                    else:
                        print("That device is not compatible")

        print("Selected Audio device: %i" % self.device_id)

    def check_device_compatibility(self, device_id: int) -> bool:
        """Check if the given audio device supports the required format.

        Args:
            device_id (int): Audio device ID.

        Returns:
            bool: True if device is compatible, False otherwise.
        """
        supported = False
        try:
            supported = self.interface.is_format_supported(
                self.rate,
                input_device=device_id,
                input_channels=self.channels,
                input_format=pyaudio.paInt16,
            )
        except Exception as e:
            print(f"Excpetion: {e}")
            supported = False
        return supported

    def list_available_devices(self) -> List[Tuple]:
        """List all available audio input devices.

        Returns:
            list: List of tuples containing (device ID, device name).

        Raises:
            Exception: If no input audio devices are available.
        """
        if not self.interface:
            self.interface = pyaudio.PyAudio()

        info = self.interface.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")
        input_devices = []

        for i in range(0, numdevices):
            device_info = self.interface.get_device_info_by_host_api_device_index(0, i)
            if device_info.get("maxInputChannels") > 0:
                input_devices.append((i, device_info.get("name")))

        if len(input_devices) == 0:
            raise Exception("There are no audio devices available")

        for i in range(0, len(input_devices)):
            print("%i --> %s" % input_devices[i])

        return input_devices

    def audio_capture_thread(self):
        """Thread function to continuously capture audio data and enqueue it.

        Detects silence and stops if no valid audio is detected for a
        threshold duration.
        """
        while not self.closed:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                zeros = bytes(self.chunk_size * 2)
                if data != zeros:
                    self.zero_counter = 0
                else:
                    self.zero_counter += 1

                if self.zero_counter > self.rate / self.chunk_size:
                    print("No audio data coming from the audio interface")
                    self.closed = True
                    break

                self.buff.put(data)

            except Exception as e:
                print(f"Error capturing audio: {e}")
                self.closed = True
                break

    def __enter__(self):
        """Open the audio stream and start the capture thread.

        Returns:
            Microphone: Self instance for use with context manager.
        """
        if not self.interface:
            self.interface = pyaudio.PyAudio()

        self.stream = self.interface.open(
            input_device_index=self.device_id,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        self.closed = False

        self.thread = threading.Thread(target=self.audio_capture_thread)
        self.thread.daemon = True
        self.thread.start()

        return self

    def __exit__(self, type, value, traceback):
        """Clean up resources: stop stream, terminate interface, join thread.

        Args:
            type (Type): Exception type.
            value (BaseException): Exception instance.
            traceback (TracebackType): Traceback object.
        """
        self.closed = True

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.interface:
            self.interface.terminate()

    def generator(self):
        """Generator that yields audio data in chunks.

        Yields:
            bytes: Concatenated audio frames.
        """
        while not self.closed:
            try:
                chunk = self.buff.get(timeout=0.5)

                if chunk is None or self.closed:
                    return

                data = [chunk]
                while True:
                    try:
                        chunk = self.buff.get(block=False)
                        if chunk is None or self.closed:
                            return
                        data.append(chunk)
                    except queue.Empty:
                        break

                yield b"".join(data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in generator: {e}")
                return


class AudioImpulseRunner(ImpulseRunner):
    """Real-time audio classifier using a microphone and Edge Impulse."""

    def __init__(self, model_path: str):
        """Initialize the AudioImpulseRunner.

        Args:
            model_path (str): Path to the Edge Impulse .eim model file.
        """
        super(AudioImpulseRunner, self).__init__(model_path)
        self.closed = True
        self.sampling_rate = 0
        self.window_size = 0
        self.labels = []

    def init(self, debug: bool | None = False) -> Dict:
        """Initialize the model and extract parameters.

        Args:
            debug (bool, optional): Whether to enable debug logging. Defaults to False.

        Returns:
            dict: Model metadata.

        Raises:
            Exception: If the model is not suitable for audio classification.
        """
        model_info = super(AudioImpulseRunner, self).init(debug)
        if model_info["model_parameters"]["frequency"] == 0:
            raise Exception(
                'Model file "'
                + self._model_path
                + '" is not suitable for audio recognition'
            )

        self.window_size = model_info["model_parameters"]["input_features_count"]
        self.sampling_rate = model_info["model_parameters"]["frequency"]
        self.labels = model_info["model_parameters"]["labels"]

        return model_info

    def __enter__(self):
        """Context manager enter method.

        Returns:
            AudioImpulseRunner: Self instance.
        """
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        """Context manager exit method. Closes the runner.

        Args:
            type (Type): Exception type.
            value (BaseException): Exception instance.
            traceback (TracebackType): Traceback object.
        """
        self.closed = True

    def classify(self, data: List) -> Dict:
        """Classifies a single audio window.

        Args:
            data (List): Audio data in int16 format.

        Returns:
            Dict: Classification results.
        """
        return super(AudioImpulseRunner, self).classify(data)

    def _process_audio_chunk(
        self, audio: bytes, features: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[Dict, bytes]]]:
        """Processes raw audio into classification results.

        Args:
            audio (bytes): Raw audio data.
            features (np.ndarray): Feature buffer from previous audio chunks.

        Returns:
            Tuple[np.ndarray, List[Tuple[Dict, bytes]]]: Updated feature buffer
                and result list.
        """
        results = []

        data = np.frombuffer(audio, dtype=np.int16)
        features = np.concatenate((features, data), axis=0)

        while len(features) >= self.window_size:
            if self.closed:
                return features, results

            try:
                res = self.classify(features[: self.window_size].tolist())
                features = features[int(self.window_size * OVERLAP) :]
                results.append((res, audio))
            except BrokenPipeError:
                self.closed = True
                return features, results
            except Exception as e:
                print(f"Error during classification: {e}")
                self.closed = True
                return features, results

        return features, results

    def _setup_microphone(self, device_id: int | None = None) -> Microphone:
        """Set up and return a microphone instance.

        Args:
            device_id (int, optional): Audio device ID to use.

        Returns:
            Microphone: Initialized microphone instance.

        Raises:
            Exception: If microphone setup fails.
        """
        return Microphone(self.sampling_rate, CHUNK_SIZE, device_id=device_id)

    def classifier(self, device_id: int | None = None):
        """Start real-time audio classification from the microphone.

        Args:
            device_id (int, optional): ID of the microphone device to use.

        Yields:
            tuple: (classification result dict, raw audio bytes)
        """
        try:
            with self._setup_microphone(device_id) as mic:
                generator = mic.generator()
                features = np.array([], dtype=np.int16)

                while not self.closed:
                    try:
                        for audio in generator:
                            if self.closed:
                                break

                            features, results = self._process_audio_chunk(
                                audio, features
                            )

                            for result in results:
                                if self.closed:
                                    return
                                yield result
                    except Exception as e:
                        print(f"Error in audio processing: {e}")
                        self.closed = True
                        return
        except Exception as e:
            print(f"Error setting up microphone: {e}")
