# Model checkpoints

Large model weights are deliberately not stored in Git. Download the default SAM 2.1 Hiera Small checkpoint from the repository root:

```bash
bash scripts/download_models.sh
```

Choose another supported model size with `SAM2_MODEL=tiny`, `small`, `base_plus`, or `large`. The checkpoint path and matching SAM 2 configuration identifier are recorded in [`../configs/model_paths.yaml`](../configs/model_paths.yaml); update both when changing sizes. The downloader accepts `SAM2_CHECKPOINT_DIR` to use another destination.

The downloader obtains official SAM 2.1 checkpoints from Meta's public files host. It downloads to a temporary file and only moves the file into place after a successful transfer. The upstream host does not publish a checksum in the project setup, so the script does not claim checksum verification.

Grounding DINO and Whisper weights are downloaded by their respective model libraries on first use and stored in external caches. They are not kept in this directory.
