# Question 3 - resultats de l'experience

Resolution : 393216 pixels. Les tailles sont celles des fichiers PNG mesures sur disque.
Le gain compare chaque PNG reconstruit au fichier PNG source correspondant. La quantification represente ici l'intensite de compression; la taille du PNG reconstruit n'est pas celle d'un fichier JPEG compresse.

| Image | Niveau | Taille PNG reelle (octets) | Taille source (octets) | Gain vs source | MSE | PSNR (dB) | Coefficients non nuls |
|---|---|---:|---:|---:|---:|---:|
| image_A.png | 420_seul | 548170 | 557596 | 1.69 % | 1.805 | 45.57 | — |
| image_A.png | faible | 530642 | 557596 | 4.83 % | 3.762 | 42.38 | 166240 |
| image_A.png | moyenne | 396599 | 557596 | 28.87 % | 16.334 | 36.00 | 40986 |
| image_A.png | forte | 192423 | 557596 | 65.49 % | 74.608 | 29.40 | 15237 |
| image_B.png | 420_seul | 897229 | 822712 | -9.06 % | 3.091 | 43.23 | — |
| image_B.png | faible | 909917 | 822712 | -10.60 % | 7.424 | 39.42 | 343899 |
| image_B.png | moyenne | 857152 | 822712 | -4.19 % | 109.624 | 27.73 | 117215 |
| image_B.png | forte | 575790 | 822712 | 30.01 % | 339.348 | 22.82 | 35360 |
| image_C.png | 420_seul | 607521 | 617995 | 1.69 % | 3.553 | 42.63 | — |
| image_C.png | faible | 619865 | 617995 | -0.30 % | 6.339 | 40.11 | 223003 |
| image_C.png | moyenne | 472804 | 617995 | 23.49 % | 29.964 | 33.36 | 51896 |
| image_C.png | forte | 164546 | 617995 | 73.37 % | 92.201 | 28.48 | 15820 |

## Reponses sur la qualite de reconstruction

Quand l'intensite de quantification augmente (faible, moyenne, forte), le PSNR diminue et l'erreur quadratique moyenne (MSE) augmente : la reconstruction s'eloigne donc de l'originale.
- **image_A.png** — PSNR (faible : 42.38 dB, moyenne : 36.00 dB, forte : 29.40 dB).
- **image_B.png** — PSNR (faible : 39.42 dB, moyenne : 27.73 dB, forte : 22.82 dB).
- **image_C.png** — PSNR (faible : 40.11 dB, moyenne : 33.36 dB, forte : 28.48 dB).
La tendance est la meme pour les trois images, mais l'ampleur de la degradation varie selon leur contenu : l'image B perd le plus de PSNR entre les niveaux faible et forte, tandis que les images A et C en perdent moins. Les planches comparatives permettent de verifier visuellement ces differences.

## Effet du sous-echantillonnage 4:2:0 seul

Y conserve 393216 echantillons; Cb et Cr en conservent chacun un quart. Le total passe de 1179648 echantillons RGB a 589824 echantillons 4:2:0, soit 50 % de moins. Cela concerne le nombre d'echantillons; les tailles de fichiers PNG du tableau sont mesurees sur disque.

## Exemples visuels

`tableau_comparatif.png` aligne les trois images sources et leurs reconstructions faible, moyenne et forte. Chaque planche *_comparaison.png ajoute aussi le cas 4:2:0 seul. Les reconstructions individuelles sont fournies.

Une baisse de la taille PNG indique une reduction du fichier par rapport a l'original; une baisse du PSNR indique une degradation de qualite.
