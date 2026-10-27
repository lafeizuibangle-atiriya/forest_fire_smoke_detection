# 本地环境说明

> 记录本机环境，方便以后重装或排查。仅本地参考。

## 版本

- Python：**3.12.10**（`D:\Soft\Python\Python312\python.exe`）
- 虚拟环境：项目下 `.venv`（已加入 `.gitignore`，不要提交）
- 系统：Windows

## 激活环境

```powershell
cd D:\Code\forest_fire_smoke_detection
.\.venv\Scripts\Activate.ps1
```

## 显卡

| 项 | 值 |
|----|----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8 GB（8188 MiB） |
| 算力 | sm_89（Ada Lovelace） |
| 驱动 | 616.56（CUDA UMD 13.4，向下兼容 12.1） |
| torch CUDA 构建 | 12.1 |
| cuDNN | 9.1.0 |

## 镜像与代理

- pip 镜像：阿里云 `https://mirrors.aliyun.com/pypi/simple/`（用 `pip config --site set` 写入）
- **代理坑**：本机 `HTTP_PROXY/HTTPS_PROXY` 曾指向失效的 `127.0.0.1:4279`，会让 pip 报
  `ProxyError ... 502 Bad Gateway`。遇到时先清掉再装：

```powershell
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''
```

- **pip.ini 不能手写**：带 UTF-8 BOM 会报 `File contains no section headers`（exit=2），
  必须用 `pip config --site set` 生成。

## 已安装关键版本

| 包 | 版本 |
|----|------|
| torch | **2.4.1+cu121** |
| torchvision | **0.19.1+cu121** |
| ultralytics | 8.3.40 |
| opencv-python | 4.10.0.84 |
| numpy | 1.26.4 |
| fastapi | 0.115.4 |
| pydantic | 2.9.2 |

## CUDA 版 PyTorch 怎么装的（重装照这个来）

**关键前提：PyPI 上 Windows 的 `torch` 是纯 CPU 构建**，`pip install torch` 装多少次
`torch.cuda.is_available()` 都是 `False`。CUDA 构建只在 PyTorch 自己的仓库里。

本机实测的下载源速度（2026-10，单连接）：

| 源 | 速度 | 结论 |
|----|------|------|
| `download.pytorch.org`（→ R2 CDN） | ≈ 0 | 不可用 |
| `mirrors.aliyun.com/pytorch-wheels/cu121` + **`-4` 强制 IPv4** | 0.5~2.7 MB/s（波动） | **可用** |
| 同上但走 IPv6 | 0.14 MB/s | 慢 5 倍以上，务必加 `-4` |
| 腾讯云 / 中科大 / 华为云 pytorch-wheels | ≈ 0 | 不可用 |

wheel 体积：`torch` **2331 MB**、`torchvision` 5.5 MB。

### 推荐做法：多线程分片下载再本地安装

单连接太慢，用 16 线程按 20MB 切片并行拉，实测 2.4~2.9 MB/s，总耗时约 20 分钟：

```powershell
# 1) 拉 wheel（只需一次；脚本思路见 git 历史里的 _dl.py）
# 2) 文件名必须符合 wheel 规范，否则 pip 报 "not a valid wheel filename"
#    torch-2.4.1+cu121-cp312-cp312-win_amd64.whl
# 3) 安装
pip install --no-deps torch-2.4.1+cu121-cp312-cp312-win_amd64.whl `
                    torchvision-0.19.1+cu121-cp312-cp312-win_amd64.whl
```

### 两个卸载/安装坑

1. **`pip install --force-reinstall` 会失败**：pip 卸载旧 torch 时把文件移到 `C:\...\Temp`，
   而 venv 在 D 盘 → `OSError [WinError 17] 系统无法将文件移到不同的磁盘驱动器`。
   而且批量删文件会被本机安全策略拦（`SAFE_DELETE_BULK_CONFIRM_REQUIRED`）。
   **解法：先用 PowerShell 直接删 `site-packages\torch`、`torchvision`、`torchgen`、
   `torch*.dist-info`，再 `pip install`（无旧版本就不会触发卸载）。**
2. 装完 `pip` 可能残留 `site-packages\~orch` 目录并报
   `WARNING: Ignoring invalid distribution ~orch`，手动删掉即可。

## 预训练权重怎么下（GitHub 拉不动）

ultralytics 默认从 `github.com/ultralytics/assets/releases` 下载，本机直连几乎不动。
用 GitHub 加速镜像：

```powershell
curl.exe -L -4 -o yolo11n.pt `
  "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
```

`ghfast.top` 实测可用（5.35MB / 16s）；`ghproxy.net`、`gh-proxy.com`、`ghp.ci` 未验证成功。

> **注意**：`yolo11n.pt` 要放在**项目根目录**。ultralytics 训练前会做 AMP 自检，
> 硬编码下载 `yolo11n.pt` 到当前工作目录；根目录已有该文件就不会再联网。
> 否则这一步在国内会卡 7 分钟以上（实测 13 KB/s）。

## 实测性能（本机）

| 项目 | 结果 |
|------|------|
| FP32 矩阵乘 4096³ | 5.47 TFLOPS |
| YOLO11n 推理 640×640 | **7.65 ms / 130.7 FPS**，显存 83 MB |
| 训练 imgsz=320 batch=4 | 显存 **0.24 GB**，约 2 it/s |
| 训练 imgsz=640 batch=16 + AMP | 预计 4~5 GB 显存（`configs/train.yaml` 默认值） |

## 常用命令

```powershell
python scripts/train.py --config configs/train.yaml
python scripts/infer.py  --config configs/infer.yaml
python scripts/serve.py  --config configs/app.yaml
pytest
ruff check . --fix
```

## 验证记录

- `torch.cuda.is_available()` = **True**，设备 `NVIDIA GeForce RTX 4060 Laptop GPU`
- 依赖导入全部通过（torch / torchvision / cv2 / ultralytics / fastapi / pydantic）
- 配置加载通过（`configs/infer.yaml`）
- `pytest`：**4 passed**
- GPU 端到端训练冒烟测试通过：`scripts/train.py` 跑通 2 epochs，日志确认
  `using GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB, sm_89)` + `AMP: checks passed`
