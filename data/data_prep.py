#!/usr/bin/env python3
import argparse
import os
import math

W, H = 28, 28
IMG_BYTES = W * H

def emit_c_array_from_bin(bin_path: str, out_path: str, var_name: str = "images"):
    size = os.path.getsize(bin_path)
    if size % IMG_BYTES != 0:
        raise SystemExit(f"File size {size} is not a multiple of {IMG_BYTES} (28*28).")

    n_images = size // IMG_BYTES

    with open(bin_path, "rb") as f, open(out_path, "w", encoding="utf-8") as out:
        out.write("// Auto-generated from raw 28x28 uint8 .bin\n")
        out.write("#pragma once\n\n")
        out.write("#include <cstddef>\n\n")
        out.write(f"static constexpr std::size_t {var_name}_N = {n_images};\n")
        out.write(f"static constexpr std::size_t {var_name}_H = {H};\n")
        out.write(f"static constexpr std::size_t {var_name}_W = {W};\n\n")

        # 3D array: [image][row][col]
        out.write(f"static const float {var_name}[{var_name}_N][{H}][{W}] = {{\n")

        # Stream and write one image at a time (keeps RAM low)
        for i in range(n_images):
            img = f.read(IMG_BYTES)
            if len(img) != IMG_BYTES:
                raise SystemExit("Unexpected EOF while reading image data.")

            out.write("  {\n")
            for r in range(H):
                out.write("    {")
                row = img[r*W:(r+1)*W]
                # normalize: byte / 255.0f [web:225]
                vals = [f"{(b / 255.0):.8f}f" for b in row]
                out.write(", ".join(vals))
                out.write("},\n")
            out.write("  },\n")

        out.write("};\n")

def main():
    ap = argparse.ArgumentParser(description="Convert raw 28x28 uint8 .bin into a C/C++ 3D float array normalized to [0,1].")
    ap.add_argument("bin_file", help="Input .bin (N images, 784 bytes each, no header)")
    ap.add_argument("-o", "--out", default=None, help="Output header file (default: <bin_basename>.h)")
    ap.add_argument("--name", default="images", help="C array variable name (default: images)")
    args = ap.parse_args()

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.bin_file))[0]
        out_path = base + ".h"

    emit_c_array_from_bin(args.bin_file, out_path, args.name)
    print(out_path)

if __name__ == "__main__":
    main()
