### Variational inference on SO(n)
This is code we used to train probabilistic rotation matrices from section 5.4

### Data
Bilingual dictionary alignment.
Dinu data: https://zenodo.org/record/2654864#.XX_GZCVS9TZ

### Preprocessing
Extracting word embeddings:
```
python bilingual.py
```

### Training
To train embeddings:
```
python procrustes_vi.py
```