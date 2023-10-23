"""Run the ScandEval benchmark."""

import os


from scandeval import benchmark


if __name__ == "__main__":
    # Set amount of threads per GPU - this is the default and is only set to prevent a
    # warning from showing
    os.environ["OMP_NUM_THREADS"] = "1"
    benchmark()
