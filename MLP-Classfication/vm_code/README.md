# VM Drive-first training

Open `train_pipeline_vm.ipynb` in the VM notebook runtime and run cells in order.

The first cell mounts Drive at `/content/drive`, fixes the dataset to `273 (lan 1)`, and assigns a unique `RUN_ID`. The next preflight cell requires a writable Drive path, PCAP files, and CUDA GPU before creating cache or running training.

All persistent files are written under:

```text
/content/drive/MyDrive/unlearning-artifacts/vm-training/experiments/<RUN_ID>/
```

Change `RUN_ID` for every concurrent VM experiment. Keep `RUN_CONTEXT='local'`; it prevents the original notebook from attempting its own Colab mount, while `DEVICE_NAME='cuda:0'` selects the VM GPU.
