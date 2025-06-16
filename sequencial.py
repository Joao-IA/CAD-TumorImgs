import os
import cv2
import numpy as np
import time

HOST_DEFINED_WARP_SIZE = 32

def save_vector(data: np.ndarray, filename: str):
    """Salva um vetor de floats em arquivo texto, 8 casas decimais."""
    with open(filename, 'w') as f:
        for v in data:
            f.write(f"{v:.8f}\n")
    print(f"  [OK] vetor salvo em {filename}")

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
    base_dir   = r"C:\Users\muril\OneDrive\Documentos\GitHub\CAD-TumorImgs\dataset_busi_drive\Dataset_BUSI_with_GT"
    output_dir = r"C:\Users\muril\OneDrive\Documentos\GitHub\CAD-TumorImgs\output_results_seq"
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    total_time = 0.0

    print("\n=== INICIANDO PROCESSAMENTO SEQUENCIAL ===")
    overall_start = time.perf_counter()

    # Agora sem filtro para "_mask"
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
            name, ext  = os.path.splitext(fname)

            img_gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"  [ERRO] falha ao ler {src_path}")
                continue
            img = img_gray.astype(np.float32) / 255.0
            h, w = img.shape

            start_img = time.perf_counter()

            # 1) Processa sequencialmente
            out_vec = process_image(img, HOST_DEFINED_WARP_SIZE)

            # 2) Salva vetor bruto
            txt_path = os.path.join(dst_folder, f"{name}_seq_output_raw.txt")
            save_vector(out_vec, txt_path)

            # 3) Reconstrói e salva imagem de saída
            out_img = reconstruct_image(out_vec, w, h, HOST_DEFINED_WARP_SIZE)
            png_path = os.path.join(dst_folder, f"{name}_haralick_out_seq.png")
            cv2.imwrite(png_path, out_img)
            print(f"  [OK] imagem salva em {png_path}")

            end_img = time.perf_counter()
            elapsed = end_img - start_img
            total_images += 1
            total_time += elapsed
            print(f"    → tempo desta imagem: {elapsed*1000:.2f} ms")

    overall_end = time.perf_counter()
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Total de imagens processadas: {total_images}")
    print(f"Tempo total processamento puro: {total_time:.3f} s")
    if total_images:
        print(f"Tempo médio por imagem: {(total_time/total_images)*1000:.2f} ms")
    print(f"Tempo geral (incluindo I/O): {(overall_end - overall_start):.3f} s")

if __name__ == "__main__":
    main()
