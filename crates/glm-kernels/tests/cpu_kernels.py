import os
from pathlib import Path
import subprocess
import tempfile

crate = Path(__file__).resolve().parents[1]
cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
with tempfile.TemporaryDirectory(prefix="glm-kernels-cpu-") as directory:
    directory = Path(directory)
    for name in ("attn.cu", "moe.cu", "tcaux.cu"):
        source = (crate / "cu" / name).read_text()
        source = source.replace('#include "common.cuh"', "")
        (directory / name).write_text(source)
    binary = directory / "cpu_kernels"
    subprocess.run([
        os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-g",
        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
        "-I" + str(cuda / "include"), "-I" + str(directory),
        str(crate / "tests" / "cpu_kernels.cpp"), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
