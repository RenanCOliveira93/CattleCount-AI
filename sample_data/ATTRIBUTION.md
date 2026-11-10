# Dados de exemplo — origem e licença

| Arquivo | O que é | Origem / licença |
|---|---|---|
| `icaerus_mauron_crop.jpg` | Recorte (2400×1800) de uma foto aérea REAL de bovinos (fazenda Mauron, França; voo DJI_202306071624_028, 100 m) | Dataset ICAERUS, Zenodo [10.5281/zenodo.11048412](https://zenodo.org/records/11048412), **CC BY 4.0**. Recortado e recomprimido (JPEG q90); metadados EXIF/XMP originais removidos pelo recorte. |
| `demo_pan_synthetic.mp4` | Vídeo **SINTÉTICO**: uma janela 1280×720 desliza sobre a foto real em resolução nativa. Os animais são estáticos; só a câmera "se move". O quadro traz o aviso "SINTETICO". | Derivado da mesma foto (CC BY 4.0). Não é vídeo aéreo real. |
| `demo_pan_synthetic_gt.txt` | Verdade exata de trilhas do vídeo acima (formato MOT: frame,id,x,y,w,h,…), derivada das caixas anotadas do dataset | Idem |

Citação do dataset: ver `docs/GUIA_TECNICO.md (seção 4)`.
