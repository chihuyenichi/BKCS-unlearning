# VM-local processed-CSV pipeline

Run `train_pipeline_vm.ipynb` from the repository root on a CUDA-capable VM. It does not mount Google Drive or parse PCAP files.

The first notebook cell is configured for these local paths:

```text
/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud       # known (0)
/home/ubuntu/Documents/KF/data_processed/icloud_100           # unknown (1)
```

Each directory must contain `train_data.csv`, `test_data.csv`, and `label_mapping.csv`. A CSV row contains `[label_id, relative_time, direction, packet_size]` repeated 10,000 times. The loader converts it to the checkpoint-compatible feature tensor `[10000,3]`.

## Run order

1. Run with `RUN_UNLEARNING=False` to index the CSV, create train/validation split, train the binary MLP, and save `base_model/best_model.pt`.
2. Inspect `run_summary.json` under `artifacts/vm-training/experiments/<RUN_ID>/`.
3. Set `RUN_UNLEARNING=True` to reuse the saved base model and run the `head_only`, `last_encoder_block`, and `full_encoder_and_head` baselines.

Raw data are read-only. Feature cache, index, checkpoint, and all outputs are local under `artifacts/`.
