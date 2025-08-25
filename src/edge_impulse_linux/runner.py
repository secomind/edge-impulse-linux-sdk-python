import json
import os.path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np


def _extract_first_json(data: bytes):
    """Extracts and returns the first JSON object found in the UTF-8 decoded data."""
    braces_open = 0
    braces_closed = 0
    line = ""
    resp = None

    for c in data.decode("utf-8"):
        if c == "{":
            line += c
            braces_open += 1
        elif c == "}":
            line += c
            braces_closed += 1
            if braces_closed == braces_open:
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    resp = None
        elif braces_open > 0:
            line += c

        if resp is not None:
            break

    return resp


def now():
    """Returns the current time in milliseconds.

    Returns:
        int: The current time in milliseconds.
    """
    return round(time.time() * 1000)


class ImpulseRunner:
    """Class to manage running an EI model.

    Manages an impulse model as a subprocess and communicaties with it
    via Unix sockets.
    """

    def __init__(self, model_path: str, timeout: int = 5):
        """Initializes the ImpulseRunner.

        Args:
            model_path (str): Path to the executable model file.
            timeout (int): timeout in seconds.
        """
        self._model_path = model_path
        self._tempdir = None
        self._runner = None
        self._client = None
        self._ix = 0
        self._debug = False
        self._hello_resp = None
        self._shm = None
        self._timeout = timeout

    def init(self, debug=False):
        """Initializes and starts the model runner subprocess.

        Args:
            debug (bool, optional): If True, runs in debug mode.
                Defaults to False.

        Returns:
            dict: The response from the initial 'hello' message.

        Raises:
            FileNotFoundError: If the model file does not exist, is not executable,
                or the runner fails to start.
        """
        if not os.path.exists(self._model_path):
            raise FileNotFoundError("Model file does not exist: " + self._model_path)

        if not os.access(self._model_path, os.X_OK):
            raise FileNotFoundError(
                'Model file "' + self._model_path + '" is not executable'
            )

        self._debug = debug
        self._tempdir = tempfile.mkdtemp()
        socket_path = os.path.join(self._tempdir, "runner.sock")
        cmd = [self._model_path, socket_path]
        if debug:
            self._runner = subprocess.Popen(cmd)
        else:
            self._runner = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        while not os.path.exists(socket_path) or self._runner.poll() is not None:
            time.sleep(0.1)

        if self._runner.poll() is not None:
            raise Exception("Failed to start runner (" + str(self._runner.poll()) + ")")

        self._client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # timeout the IPC connection in case the EIM hangs
        self._client.settimeout(self._timeout)
        self._client.connect(socket_path)

        hello_resp = self._hello_resp = self.hello()

        if "features_shm" in hello_resp.keys():
            shm_name = hello_resp["features_shm"]["name"]
            # python does not want the leading slash
            shm_name = shm_name.lstrip("/")
            shm = shared_memory.SharedMemory(name=shm_name)
            self._shm = {
                "shm": shm,
                "type": hello_resp["features_shm"]["type"],
                "elements": hello_resp["features_shm"]["elements"],
                "array": np.ndarray(
                    (hello_resp["features_shm"]["elements"],),
                    dtype=np.float32,
                    buffer=shm.buf,
                ),
            }

        return self._hello_resp

    def __del__(self):
        self.stop()

    def stop(self):
        """Stops the runner process and cleans up resources."""
        if self._tempdir is not None:
            shutil.rmtree(self._tempdir)
            self._tempdir = None

        if self._client is not None:
            self._client.close()
            self._client = None

        if self._runner is not None:
            os.kill(self._runner.pid, signal.SIGINT)
            # todo: in Node we send a SIGHUP after 0.5sec if process has not died, can we do this somehow here too?
            self._runner = None

        if self._shm is not None:
            self._shm["shm"].close()
            resource_tracker.unregister(self._shm["shm"]._name, "shared_memory")
            self._shm = None

    def hello(self):
        """Sends a 'hello' message to the runner; returns info about the model.

        Returns:
            dict: The response from the runner.
        """
        msg = {"hello": 1}
        return self.send_msg(msg)

    def classify(self, data):
        """Classifies the given data using the model.

        Args:
            data (any): The data to classify.

        Returns:
            dict: The classification response.

        Raises:
            Exception: If classification fails.
        """
        if self._shm:
            self._shm["array"][:] = data

            msg = {
                "classify_shm": {
                    "elements": len(data),
                }
            }
        else:
            msg = {"classify": data}
        if self._debug:
            msg["debug"] = True

        send_resp = self.send_msg(msg)
        return send_resp

    def set_threshold(self, obj):
        if "id" not in obj:
            raise Exception('set_threshold requires an object with an "id" field')

        msg = {"set_threshold": obj}
        return self.send_msg(msg)

    def send_msg(self, msg):
        """Classifies the given data using the model.

        Args:
            data (any): The data to classify.

        Returns:
            dict: The classification response.

        Raises:
            Exception: If classification fails.
        """
        if not self._client:
            raise Exception("ImpulseRunner is not initialized (call init())")

        self._ix = self._ix + 1
        ix = self._ix

        msg["id"] = ix
        self._client.send(json.dumps(msg).encode("utf-8"))

        data = b""
        while True:
            chunk = self._client.recv(1024)
            # end chunk has \x00 in the end
            if chunk[-1] == 0:
                data = data + chunk[:-1]
                break
            data = data + chunk

        resp = _extract_first_json(data)

        if resp is None:
            raise Exception("No data or corrupted data received")

        if resp["id"] != ix:
            raise Exception("Wrong id, expected: " + str(ix) + " but got " + resp["id"])

        if not resp["success"]:
            raise Exception(resp["error"])

        del resp["id"]
        del resp["success"]

        return resp
