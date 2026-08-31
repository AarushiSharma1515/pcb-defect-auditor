EarlyStopping: Training stopped early as no improvement observed in last 20 epochs. Best results observed at epoch 79, best model saved as best.pt.
To update EarlyStopping(patience=20) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

99 epochs completed in 0.498 hours.
Optimizer stripped from /content/runs/detect/pcb_defect/deeppcb_run-4/weights/last.pt, 6.3MB
Optimizer stripped from /content/runs/detect/pcb_defect/deeppcb_run-4/weights/best.pt, 6.3MB

Validating /content/runs/detect/pcb_defect/deeppcb_run-4/weights/best.pt...
Ultralytics 8.4.135 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
Model summary (fused): 73 layers, 3,006,818 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 1.1it/s 4.5s
                   all        150        984      0.974      0.952      0.979      0.791
                  Open        134        191      0.969       0.97      0.988      0.731
                 Short        114        170      0.981       0.89      0.948      0.703
             Mousebite        129        194      0.959      0.966      0.976      0.769
                  Spur        113        153      0.976      0.954      0.976      0.751
                Copper        119        140      0.985      0.969      0.995      0.911
              Pin-hole        124        136      0.973      0.963      0.989      0.879
Speed: 0.6ms preprocess, 5.8ms inference, 0.0ms loss, 6.4ms postprocess per image
Results saved to /content/runs/detect/pcb_defect/deeppcb_run-4