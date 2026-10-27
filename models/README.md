# models/ — 模型权重目录

按 `.gitignore` 约定，**所有 `*.pt` 权重不入库**：

- **自训练权重**（`best.pt` = v3、`best_pseudo.pt` = v5，各约 5.2 MB）：从本仓库
  **GitHub Releases 附件**下载，放入本目录。
- **预训练权重**（`yolo11n.pt`，Ultralytics AMP 自检需要，放项目根目录）：

  ```powershell
  curl.exe -L -4 -o yolo11n.pt `
    "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
  ```

权重来源说明：v3 / v5 均为 YOLO11n 在 FIgLib 官方标注（+ v5 伪标签回灌）上训练，
训练配置见 `configs/train_figlib_opt.yaml` 与 `configs/train_figlib_pseudo.yaml`，
复现命令见 README「训练与评估」一节。
