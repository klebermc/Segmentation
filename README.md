# Segmentation

> **Note:** The idea is Kleber Cabral's. The training script was generated with an AI assistant from his description, and the README documentation was added with AI assistance (Claude).

> **Untested idea.** This approach was never trained or evaluated end to end. It's published to share the idea (unsupervised pretraining of an encoder on snowy images, then a segmentation decoder on top of the frozen encoder), not as working or validated code. No results are claimed.

Trains a convolutional autoencoder, then reuses its frozen encoder to train a separate segmentation decoder — a two-stage "pretrain then adapt" approach for segmenting snowy camera images.

## Structure

```
Segmentation/
├── src/
│   └── train_segmentation_model.py
└── requirements.txt
```

## What it does

`src/train_segmentation_model.py` defines and trains, in sequence:

1. **`Encoder`** — 3 conv layers compressing a 128×128 RGB image into a 128-dim latent vector.
2. **`Decoder`** — mirrors the encoder with transpose-conv layers to reconstruct the image from the latent vector.
3. **Autoencoder training (stage 1)** — `Encoder` + `Decoder` trained together for 20 epochs with MSE reconstruction loss on snowy images, so the encoder learns useful features unsupervised.
4. **`SegmentationDecoder`** — same architecture shape as `Decoder`, but outputs per-pixel class probabilities (softmax over `num_classes`) instead of a reconstructed image.
5. **Segmentation training (stage 2)** — the encoder is frozen (`encoder.eval()`), and only `SegmentationDecoder` is trained for 20 epochs with cross-entropy loss against mask images.

A custom `SnowyDataset` (`torch.utils.data.Dataset`) loads images (and optional masks) from a directory, resizing to 128×128 and normalizing with ImageNet statistics.

**Note on the data:** the script currently uses "clear" images from `cam6/small_clear` (loaded as grayscale) as if they were segmentation masks for the corresponding snowy images in `cam6/small_snow`, rather than real per-pixel class-label masks — this looks like a simplification/placeholder rather than a proper labeled segmentation dataset.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/train_segmentation_model.py
```

**Requires data that isn't included in this project:** the script expects `cam6/small_snow/` and `cam6/small_clear/` directories (relative to wherever you run it from) containing the training images. It will not run without supplying these — either create them at those relative paths or edit the `image_dir`/`mask_dir` values near the bottom of the script.

## Key dependencies

- `torch`, `torchvision`
- `pillow`

## Status

**Untested; idea only** (see the note at the top). Study/teaching-style code (heavily line-commented) rather than a production training pipeline. Runs top-to-bottom as a script with no `if __name__ == "__main__":` guard, no CLI arguments, no model checkpoint saving, and no evaluation/validation split — hyperparameters (epochs, learning rate, batch size, latent dim) are hardcoded inline. Not runnable without supplying the expected data directories.

## License

MIT — see [LICENSE](LICENSE).
