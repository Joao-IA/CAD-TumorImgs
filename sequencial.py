import os
import cv2
import numpy as np
import time

HOST_DEFINED_WARP_SIZE = 32

def save_vector(data: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        for v in data:
            f.write(f"{v:.8f}\n")

def process_image(img: np.ndarray, warp_size: int = HOST_DEFINED_WARP_SIZE) -> np.ndarray:
    h, w = img.shape
    total = h * w
    out_size = (total + warp_size - 1) // warp_size
    out = np.zeros(out_size, dtype=np.float32)
    for wi in range(out_size):
        acc = 0.0
        base = wi * warp_size
        for k in range(warp_size):
            idx = base + k
            if idx >= total:
                break
            y, x = divmod(idx, w)
            c = img[y, x]
            r = img[y, x + 1] if x + 1 < w else c
            acc += 0.5 * (c + r)
        out[wi] = acc
    return out

def reconstruct_image(out_vec: np.ndarray, w: int, h: int, warp_size: int = HOST_DEFINED_WARP_SIZE) -> np.ndarray:
    new_w = max(w // warp_size, 1)
    mat = np.zeros((h, new_w), dtype=np.float32)
    for r in range(h):
        for c in range(new_w):
            idx = r * new_w + c
            if idx < out_vec.size:
                mat[r, c] = out_vec[idx]
    norm = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

def main():
    base_dir   = "/dataset_busi_drive/Dataset_BUSI_with_GT"
    output_dir = "/output_results_seq"
    os.makedirs(output_dir, exist_ok=True)

    total_calc_time = 0.0
    image_count = 0  

    print("\n ---Iniciando processamento sequencial---")

    for label in sorted(os.listdir(base_dir)):
        src_folder = os.path.join(base_dir, label)
        if not os.path.isdir(src_folder):
            continue

        dst_folder = os.path.join(output_dir, label)
        os.makedirs(dst_folder, exist_ok=True)

        print(f"\nProcessando pasta '{label}':")
        for fname in sorted(os.listdir(src_folder)):
            if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg")):
                continue

            src_path = os.path.join(src_folder, fname)
            name, _  = os.path.splitext(fname)

            img_gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"  [ERRO] falha ao ler {src_path}")
                continue

            img = img_gray.astype(np.float32) / 255.0
            h, w = img.shape

            start = time.perf_counter()
            out_vec = process_image(img, HOST_DEFINED_WARP_SIZE)
            end = time.perf_counter()

            elapsed = end - start
            total_calc_time += elapsed
            image_count += 1

            print(f"  → '{fname}' cálculo: {elapsed*1000:.2f} ms")

            save_vector(out_vec, os.path.join(dst_folder, f"{name}_seq_output_raw.txt"))
            out_img = reconstruct_image(out_vec, w, h, HOST_DEFINED_WARP_SIZE)
            cv2.imwrite(os.path.join(dst_folder, f"{name}_haralick_out_seq.png"), out_img)

    print("\n---Processamento concluído---")
    print(f"Tempo total de execução para {image_count} imagens: {total_calc_time:.6f} s")
    if image_count > 0:
        avg = total_calc_time / image_count
        print(f"Média por imagem: {avg*1000:.5f} ms")

if __name__ == "__main__":
    main()
