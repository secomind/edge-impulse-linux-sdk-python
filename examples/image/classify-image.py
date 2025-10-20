#!/usr/bin/env python

# Device specific patches for Jetson Nano (needs to be before importing cv2)
from device_patches import initialize_device

initialize_device()
try:
    import cv2
except ImportError:
    print('Missing OpenCV, install via `pip3 install "opencv-python>=4.5.1.48,<5"`')
    exit(1)
import getopt
import os
import sys

from edge_impulse_linux.image import ImageImpulseRunner

runner = None


def help():
    print("python classify-image.py <path_to_model.eim> <path_to_image.jpg>")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            help()
            sys.exit()

    if len(args) != 2:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print("MODEL: " + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            # model_info = runner.init(debug=True, timeout=10) # to get debug print out and set longer timeout

            print(
                'Loaded runner for "'
                + model_info["project"]["owner"]
                + " / "
                + model_info["project"]["name"]
                + '"'
            )
            labels = model_info["model_parameters"]["labels"]

            img = cv2.imread(args[1])
            if img is None:
                print("Failed to load image", args[1])
                exit(1)

            # imread returns images in BGR format, so we need to convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # get_features_from_image also takes a crop direction arguments in case you don't have square images
            # features, cropped = runner.get_features_from_image(img)

            # this mode uses the same settings used in studio to crop and resize the input
            features, cropped = runner.get_features_from_image_auto_studio_settings(img)

            res = runner.classify(features)

            if "classification" in res["result"].keys():
                print(
                    "Result (%d ms.) "
                    % (res["timing"]["dsp"] + res["timing"]["classification"]),
                    end="",
                )
                for label in labels:
                    score = res["result"]["classification"][label]
                    print("%s: %.2f\t" % (label, score), end="")
                print("", flush=True)

            elif "bounding_boxes" in res["result"].keys():
                print(
                    "Found %d bounding boxes (%d ms.)"
                    % (
                        len(res["result"]["bounding_boxes"]),
                        res["timing"]["dsp"] + res["timing"]["classification"],
                    )
                )
                for bb in res["result"]["bounding_boxes"]:
                    print(
                        "\t%s (%.2f): x=%d y=%d w=%d h=%d"
                        % (
                            bb["label"],
                            bb["value"],
                            bb["x"],
                            bb["y"],
                            bb["width"],
                            bb["height"],
                        )
                    )
                    cropped = cv2.rectangle(
                        cropped,
                        (bb["x"], bb["y"]),
                        (bb["x"] + bb["width"], bb["y"] + bb["height"]),
                        (255, 0, 0),
                        1,
                    )

            elif "freeform" in res["result"].keys():
                print(
                    "Result (%d ms.)"
                    % (res["timing"]["dsp"] + res["timing"]["classification"])
                )
                for i in range(0, len(res["result"]["freeform"])):
                    print(
                        f"    Freeform output {i}:",
                        ", ".join(f"{x:.4f}" for x in res["result"]["freeform"][i]),
                    )

            if "visual_anomaly_grid" in res["result"].keys():
                print(
                    "Found %d visual anomalies (%d ms.)"
                    % (
                        len(res["result"]["visual_anomaly_grid"]),
                        res["timing"]["dsp"]
                        + res["timing"]["classification"]
                        + res["timing"]["anomaly"],
                    )
                )
                for grid_cell in res["result"]["visual_anomaly_grid"]:
                    print(
                        "\t%s (%.2f): x=%d y=%d w=%d h=%d"
                        % (
                            grid_cell["label"],
                            grid_cell["value"],
                            grid_cell["x"],
                            grid_cell["y"],
                            grid_cell["width"],
                            grid_cell["height"],
                        )
                    )
                    cropped = cv2.rectangle(
                        cropped,
                        (grid_cell["x"], grid_cell["y"]),
                        (
                            grid_cell["x"] + grid_cell["width"],
                            grid_cell["y"] + grid_cell["height"],
                        ),
                        (255, 125, 0),
                        1,
                    )
                values = [
                    grid_cell["value"]
                    for grid_cell in res["result"]["visual_anomaly_grid"]
                ]
                mean_value = sum(values) / len(values)
                max_value = max(values)
                print("Max value: %.2f" % max_value)
                print("Mean value: %.2f" % mean_value)

            # the image will be resized and cropped, save a copy of the picture here
            # so you can see what's being passed into the classifier
            cv2.imwrite("debug.jpg", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        finally:
            if runner:
                runner.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
