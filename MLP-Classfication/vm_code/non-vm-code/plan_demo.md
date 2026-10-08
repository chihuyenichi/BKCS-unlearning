# Plan — VM-local CSV classification and unlearning

This plan describes [train_pipeline_vm.ipynb](train_pipeline_vm.ipynb), a group-wise unlearning workflow running on VM-local CSV data.

## Data source

The VM reads only local processed CSV files:

```text
known AOL:    /home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud
unknown 273:  /home/ubuntu/Documents/KF/data_processed/icloud_100
```

Each source contains `train_data.csv`, `test_data.csv`, and `label_mapping.csv`.
`label_mapping.csv` maps `label_name` to `label_id`. Each data row has 40,000 numeric values and no header:

```text
[label_id, relative_time, direction, packet_size] × 10,000
```

The label is repeated per packet. The loader reshapes each row to `[10000,4]`, validates the label column, then removes it to give the model `[10000,3] = [relative_time, direction, raw_packet_size]`.

## Split protocol

- `test_data.csv` is a fixed final test set and is never used to select parameters.
- `train_data.csv` is split deterministically by `(source, original_label)` into 85% train and 15% validation.
- AOL records always receive binary target `known=0`; `icloud_100` records always receive `unknown=1`.
- Original labels are retained for class-level forgetting.

Validation is required to choose the unknown threshold, base-training epoch, and unlearning epoch without contaminating final test results.

## Resource behaviour

The supplied AOL CSV files are about 941–942 MB each; the unknown CSV files are about 1.9 GB each. The notebook first makes a byte-offset index, then seeks and parses one row per sample instead of loading the whole CSV into RAM. An optional local tensor cache accelerates later epochs:

```text
./artifacts/vm-training/experiments/<RUN_ID>/
├── csv_index.json
├── split_manifest.json
├── feature_cache/
├── base_model/best_model.pt
├── base_model/README.md
├── run_summary.json
└── unlearning/
```

Raw CSV files are read-only and all output remains local to the VM.

## Model and unlearning

`MLP-Classfication/vm_code/weight_encoder_trained/pretrain_AOL.pth` is loaded strictly into `RawPacketEncoder`. Its expected raw input is exactly `[10000,3]`. A binary MLP is trained on both AOL and 273 source records. The path configuration detects the repository root, including when Jupyter starts in the notebook folder.

For a forgotten AOL folder label `B`:

```text
Df = AOL samples with original label B
Dr = all remaining AOL samples + all 273 samples
loss = CE(Df, unknown=1) + lambda * CE(Dr, original binary label)
```

The optional phase runs three update-scope baselines: `head_only`, `last_encoder_block`, and `full_encoder_and_head`. Each output is saved separately under `unlearning/`.

## Run

1. Clone the repository on the GPU VM.
2. Confirm the two paths in the first notebook cell; they are already set to the supplied paths.
3. Run the notebook with `RUN_UNLEARNING=False` to make the base model.
4. Inspect `run_summary.json` and keep test results untouched.
5. Set `RUN_UNLEARNING=True`, configure `FORGET_LABEL_COUNTS` and `FORGET_LABEL_SEED`, then run again to compare the three update scopes. The saved run used nested groups of 1, 4, and 8 labels, with 3 repeats each.

The default output is resolved under `<repo>/artifacts/vm-training/experiments/<RUN_ID>/`. Use a distinct `RUN_ID` for each experiment to keep its artifacts separate.
