"""Experimentation DCT/4:2:0 pour le travail pratique INF8770.

Les tailles rapportees sont les tailles reelles sur disque des PNG reconstruits,
comparees a celles des PNG sources. Elles ne sont pas des tailles JPEG.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom


IMAGE_FILES = ("image_A.png", "image_B.png", "image_C.png")
BLOCK_SIZE = 8

# Matrice JPEG luminance de reference, utilisee pour les trois composantes.
BASE_QUANTIZATION = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)

QUANTIZATION_SCALES = {
    "faible": 0.5,
    "moyenne": 1.0,
    "forte": 2.0,
}


def quantization_matrix(scale: float) -> np.ndarray:
    """Construit une matrice positive a partir de la matrice de reference."""
    return np.maximum(1, np.floor(BASE_QUANTIZATION * scale + 0.5))


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convertit un tableau RGB 8 bits en composantes Y, Cb et Cr flottantes."""
    rgb_float = rgb.astype(np.float64)
    red, green, blue = np.moveaxis(rgb_float, -1, 0)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return y, cb, cr


def ycbcr_to_rgb(
    y: np.ndarray, cb: np.ndarray, cr: np.ndarray
) -> np.ndarray:
    """Convertit des composantes YCbCr flottantes en RGB borne entre 0 et 255."""
    cb_centered = cb - 128.0
    cr_centered = cr - 128.0
    red = y + 1.402 * cr_centered
    green = y - 0.344136 * cb_centered - 0.714136 * cr_centered
    blue = y + 1.772 * cb_centered
    rgb = np.stack((red, green, blue), axis=-1)
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)


def subsample_420(chroma: np.ndarray) -> np.ndarray:
    """Reduit une chrominance par moyenne de chaque voisinage 2 x 2."""
    height, width = chroma.shape
    if height % 2 or width % 2:
        raise ValueError("Le sous-echantillonnage 4:2:0 exige des dimensions paires.")
    return chroma.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


def upsample_420(chroma: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reagrandit la chrominance par interpolation bilineaire."""
    upsampled = zoom(chroma, (2, 2), order=1, mode="nearest", prefilter=False)
    if upsampled.shape != shape:
        raise ValueError(
            f"Dimensions chroma reconstruites {upsampled.shape}, attendu {shape}."
        )
    return upsampled


def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Arrondit au plus proche, les egalites s'eloignant de zero."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def transform_blocks(plane: np.ndarray) -> np.ndarray:
    """Applique une DCT-II orthonormale a tous les blocs 8 x 8 du plan."""
    height, width = plane.shape
    if height % BLOCK_SIZE or width % BLOCK_SIZE:
        raise ValueError("Chaque plan doit avoir des dimensions multiples de 8.")
    blocks = plane.reshape(
        height // BLOCK_SIZE, BLOCK_SIZE, width // BLOCK_SIZE, BLOCK_SIZE
    ).transpose(0, 2, 1, 3)
    return dctn(blocks - 128.0, type=2, norm="ortho", axes=(-2, -1))


def inverse_transform_blocks(coefficients: np.ndarray) -> np.ndarray:
    """Applique la DCT inverse et recompose le plan."""
    blocks = idctn(coefficients, type=2, norm="ortho", axes=(-2, -1)) + 128.0
    block_rows, block_columns = blocks.shape[:2]
    return (
        blocks.transpose(0, 2, 1, 3)
        .reshape(block_rows * BLOCK_SIZE, block_columns * BLOCK_SIZE)
    )


def quantize_and_reconstruct(
    plane: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Quantifie les coefficients du plan puis reconstruit celui-ci."""
    coefficients = transform_blocks(plane)
    quantized = round_half_away_from_zero(coefficients / matrix)
    reconstructed = inverse_transform_blocks(quantized * matrix)
    return reconstructed, quantized


def compress_reconstruct(
    rgb: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Applique la chaine YCbCr, 4:2:0, DCT, quantification et reconstruction."""
    height, width, _ = rgb.shape
    if height % 16 or width % 16:
        raise ValueError(
            "La hauteur et la largeur doivent etre multiples de 16 "
            "pour les blocs 8 x 8 en 4:2:0."
        )

    y, cb, cr = rgb_to_ycbcr(rgb)
    cb_small = subsample_420(cb)
    cr_small = subsample_420(cr)

    y_reconstructed, y_quantized = quantize_and_reconstruct(y, matrix)
    cb_reconstructed, cb_quantized = quantize_and_reconstruct(cb_small, matrix)
    cr_reconstructed, cr_quantized = quantize_and_reconstruct(cr_small, matrix)

    cb_full = upsample_420(cb_reconstructed, (height, width))
    cr_full = upsample_420(cr_reconstructed, (height, width))
    reconstructed_rgb = ycbcr_to_rgb(y_reconstructed, cb_full, cr_full)

    all_quantized = np.concatenate(
        (y_quantized.ravel(), cb_quantized.ravel(), cr_quantized.ravel())
    )
    return reconstructed_rgb, all_quantized


def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    difference = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(np.square(difference)))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = mean_squared_error(original, reconstructed)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def save_comparison_sheet(path: Path, original: np.ndarray, reconstructions: dict[str, np.ndarray]) -> None:
    """Cree une planche cote a cote pour comparer les degradations visuellement."""
    from PIL import ImageDraw
    labels = ["Originale", *reconstructions.keys()]
    images = [Image.fromarray(original, mode="RGB")]
    images.extend(Image.fromarray(image, mode="RGB") for image in reconstructions.values())
    thumb_width = 320
    thumb_height = max(1, round(original.shape[0] * thumb_width / original.shape[1]))
    label_height = 36
    canvas = Image.new("RGB", (thumb_width * len(images), thumb_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * thumb_width
        draw.text((x + 8, 10), label, fill="black")
        canvas.paste(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, label_height))
    canvas.save(path)


def save_all_images_comparison(
    path: Path,
    originals: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Cree une grille commune des originaux et des trois niveaux de quantification."""
    from PIL import ImageDraw

    levels = ["Originale", *QUANTIZATION_SCALES]
    cell_width = 320
    header_height = 36
    row_label_width = 100
    first_image = next(iter(originals.values()))
    thumb_height = max(
        1, round(first_image.shape[0] * cell_width / first_image.shape[1])
    )
    row_height = thumb_height + header_height
    canvas = Image.new(
        "RGB",
        (row_label_width + cell_width * len(levels), row_height * len(originals)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for row_index, (filename, original) in enumerate(originals.items()):
        stem = Path(filename).stem
        top = row_index * row_height
        draw.text((8, top + 10), stem, fill="black")
        for column_index, level in enumerate(levels):
            left = row_label_width + column_index * cell_width
            draw.text((left + 8, top + 10), level, fill="black")
            if level == "Originale":
                image = Image.fromarray(original, mode="RGB")
            else:
                image_path = output_dir / f"{stem}_{level}.png"
                with Image.open(image_path) as reconstructed:
                    image = reconstructed.convert("RGB")
            canvas.paste(
                image.resize(
                    (cell_width, thumb_height), Image.Resampling.LANCZOS
                ),
                (left, top + header_height),
            )
    canvas.save(path)

def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "resultats_tp2"
    output_dir.mkdir(exist_ok=True)

    images: dict[str, np.ndarray] = {}
    source_file_sizes: dict[str, int] = {}
    for filename in IMAGE_FILES:
        image_path = root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Image requise introuvable : {image_path}")
        source_file_sizes[filename] = image_path.stat().st_size
        with Image.open(image_path) as image:
            images[filename] = np.asarray(image.convert("RGB"), dtype=np.uint8)

    dimensions = {image.shape for image in images.values()}
    if len(dimensions) != 1:
        raise ValueError("Les trois images doivent avoir les memes dimensions.")

    height, width, _ = next(iter(images.values())).shape
    pixel_count = height * width
    result_rows: list[dict[str, object]] = []

    matrices_path = output_dir / "matrices_quantification.csv"
    with matrices_path.open("w", newline="", encoding="utf-8-sig") as matrix_file:
        matrix_writer = csv.writer(matrix_file, delimiter=";")
        matrix_writer.writerow(["niveau", *[f"c{i}" for i in range(1, 9)]])
        for level, scale in QUANTIZATION_SCALES.items():
            matrix = quantization_matrix(scale).astype(int)
            for row_number, row in enumerate(matrix, start=1):
                matrix_writer.writerow(
                    [f"{level}_ligne_{row_number}", *row.tolist()]
                )

    print(f"Dimensions: {width} x {height}")
    print("Matrices de quantification (identiques pour Y, Cb et Cr):")
    for level, scale in QUANTIZATION_SCALES.items():
        print(f"\n{level} (facteur {scale:g}):")
        print(quantization_matrix(scale).astype(int))

    for filename, original in images.items():
        stem = Path(filename).stem
        y, cb, cr = rgb_to_ycbcr(original)
        cb_420 = subsample_420(cb)
        cr_420 = subsample_420(cr)
        chroma_only = ycbcr_to_rgb(
            y,
            upsample_420(cb_420, (height, width)),
            upsample_420(cr_420, (height, width)),
        )

        if filename:
            output_path = output_dir / f"{stem}_420_seul.png"
            save_rgb(output_path, chroma_only)
            output_file_size = output_path.stat().st_size
            source_file_size = source_file_sizes[filename]
            result_rows.append(
                {
                    "image": filename,
                    "niveau": "420_seul",
                    "mse": mean_squared_error(original, chroma_only),
                    "psnr_db": psnr(original, chroma_only),
                    "coefficients_non_nuls": "",
                    "nombre_coefficients": "",
                    "echantillons_420": int(pixel_count * 1.5),
                    "taille_fichier_octets": output_file_size,
                    "taille_source_octets": source_file_size,
                    "gain_vs_original_pct": 100.0
                    * (1.0 - output_file_size / source_file_size),
                }
            )

        for level, scale in QUANTIZATION_SCALES.items():
            reconstructed, quantized = compress_reconstruct(
                original, quantization_matrix(scale)
            )
            output_path = output_dir / f"{stem}_{level}.png"
            save_rgb(output_path, reconstructed)

            output_file_size = output_path.stat().st_size
            source_file_size = source_file_sizes[filename]
            nonzero_count = int(np.count_nonzero(quantized))
            result_rows.append(
                {
                    "image": filename,
                    "niveau": level,
                    "mse": mean_squared_error(original, reconstructed),
                    "psnr_db": psnr(original, reconstructed),
                    "coefficients_non_nuls": nonzero_count,
                    "nombre_coefficients": int(quantized.size),
                    "echantillons_420": int(pixel_count * 1.5),
                    "taille_fichier_octets": output_file_size,
                    "taille_source_octets": source_file_size,
                    "gain_vs_original_pct": 100.0
                    * (1.0 - output_file_size / source_file_size),
                }
            )
            print(
                f"{filename} | {level:7s} | "
                f"PSNR={result_rows[-1]['psnr_db']:.2f} dB | "
                f"coefficients non nuls={nonzero_count}/{quantized.size} | "
                f"taille PNG={output_file_size} octets"
            )

    results_path = output_dir / "resultats.csv"
    fieldnames = list(result_rows[0])
    with results_path.open("w", newline="", encoding="utf-8-sig") as results_file:
        writer = csv.DictWriter(
            results_file, fieldnames=fieldnames, delimiter=";"
        )
        writer.writeheader()
        writer.writerows(result_rows)

    for filename, original in images.items():
        stem = Path(filename).stem
        # reconstructions = {"4:2:0 seul": np.asarray(Image.open(output_dir / f"{stem}_420_seul.png"))}
        reconstructions = {}
        for level in QUANTIZATION_SCALES:
            reconstructions[level] = np.asarray(Image.open(output_dir / f"{stem}_{level}.png"))
        save_comparison_sheet(output_dir / f"{stem}_comparaison.png", original, reconstructions)

    comparison_path = output_dir / "tableau_comparatif.png"
    save_all_images_comparison(comparison_path, images, output_dir)

    print(f"\nMesures enregistrees dans : {results_path}")
    print(f"Resume Question 3 : {output_dir / 'rapport_question3.md'}")
    print(f"Planches comparatives : {output_dir}/*_comparaison.png")
    print(f"Tableau comparatif des trois images : {comparison_path}")
    print(f"Images reconstruites dans : {output_dir}")


if __name__ == "__main__":
    main()