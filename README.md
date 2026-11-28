# DeepCreamPy 3.0 (Unofficial)

[DeepCreamPy 2.2.0 BETA](https://github.com/Deepshift/DeepCreamPy)をPyTorch、PySide6で書き直した非公式アップデート版です。Windows 11 x64, Python 3.13.17, uv 0.13.5で動作確認をしました。

## (Windowsユーザー向け) ビルド済みexeファイルを使用する

[最新リリース](https://github.com/3kanAlpha/DeepCreamPy-torch/releases/latest)から `DeepCreamPy-3.0.x-Windows-x64.7z`（Anima対応版を利用する場合は `DeepCreamPy-Anima-Windows-x64.7z`）をダウンロードし、7zを展開した後 `DeepCreamPy.exe` を起動してください。Pythonやuvのインストールは不要です。

Windows以外を利用している方や、ソースから直接実行したい方は以下の手順を参照してください。

## セットアップ手順

[uv](https://docs.astral.sh/uv/getting-started/installation/)をインストールしていない場合は、公式サイトからインストールしてください。

uvのインストール後、以下を実行します。

```sh
git clone https://github.com/3kanAlpha/DeepCreamPy-torch.git
cd DeepCreamPy-torch
uv sync --locked
```

CUDA 13.2対応のPyTorchを含む必要なパッケージがインストールされます。

Anima（+ ControlNet-LLLite）を使用する場合は、オプション依存グループ `anima` を含めて同期してください（CUDA対応NVIDIA GPU専用）。

```sh
uv sync --locked --group anima
```

## 画像の復元に必要なモデルの準備

以前のバージョンのDeepCreamPy用のモデルを持っていない場合、以下からダウンロードしてください。

[original TensorFlow model archive](https://drive.google.com/open?id=1IMwzqZUuRnTv5jcuKdvZx-RZweknww5x)

ダウンロードしたモデルはそれぞれ`models/bar/`と`models/mosaic/`へ配置してください。それぞれのフォルダに`checkpoint`, `.index`, `.data-*`という名前のファイルがあることを確認してください。

以下のコマンドを実行して、TensorFlow用モデルからPyTorch用モデルを生成します。

```sh
uv run --locked --with tensorflow==2.21.0 python src/convert_models.py
```

このコマンドで`models/bar/inpaint.pt`と`models/mosaic/inpaint.pt`が生成されます。変換元のモデルファイルも保持されます。（オリジナルが必要ない場合は削除しても問題ありません）

### Anima (+ ControlNet-LLLite) を使用する場合（Bar モード・GPU 専用）

CUDA対応のNVIDIA GPU（VRAM 8GB クラスの環境では **FP8 ストレージ量子化** が既定で有効になり、ブロック単位オフロードなしで高速動作します。詳細設定や `--no-anima-fp8` でオフにも切り替え可能）を使用している場合、Barモードの復元モデルとして **Anima (`Anima-Base` + 4chインペイント用 ControlNet-LLLite)** を選択できます。

Animaのモデルウェイトは非商用ライセンスのため配布物には同梱されていません。[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) および [kohya-ss/Anima-LLLite](https://huggingface.co/kohya-ss/Anima-LLLite) 等からComfyUI互換の単体 `.safetensors` ファイルを入手し、以下の4フォルダへ配置してください（ファイル名のリネームは不要です）。

| フォルダ | 配置するファイル例 |
| --- | --- |
| `models/anima/diffusion/` | `anima-base-v1.0.safetensors`（Anima-Base本体） |
| `models/anima/text_encoder/` | `qwen_3_06b_base.safetensors`（Qwen3-0.6B） |
| `models/anima/vae/` | `qwen_image_vae.safetensors`（Qwen-Image VAE） |
| `models/anima/controlnet/` | `anima-lllite-inpainting-v2.safetensors`（4chインペイント用 ControlNet-LLLite） |

* 推論に必要なトークナイザ定義は `assets/anima_tokenizers/` に同梱されているため、ウェイト配置後は完全オフラインで動作します。
* `diffusion/` や `controlnet/` に複数の `.safetensors` を配置した場合は、GUIの「Anima Settings → Advanced Settings」内のドロップダウンから切り替えられます。
* 補助プロンプト・VRAM対策・詳細パラメータについては [モデル解説（日本語）](docs/MODELS.ja.md#anima-モデル--4ch-インペイント-controlnet-llliteバーモードgpu-専用) を参照してください。

## ツールの起動

以下のコマンドでGUIが起動します。

```sh
uv run --locked python src/main.py
```

Animaを使用する場合は `--group anima` を付けて起動してください。

```sh
uv run --locked --group anima python src/main.py
```

### GUI上でマスク編集から復元まで行う（Mask & Decensor）

外部の画像編集ソフトで事前に緑色マスクを作成しなくても、メイン画面の **Mask & Decensor** ボタンから内蔵エディタを開いてその場でマスク作成と復元を行えます。

1. メイン画面の **Mask & Decensor** をクリックしてエディタウィンドウを開く（モデル読み込み中でも起動できます）。
2. 画像をドラッグ＆ドロップ（または **Open Image...** / `Ctrl+V` で貼り付け）して開く。
3. **Auto Detect** ツールで黒消し・白消しの周辺を矩形選択すると、欠損箇所が自動で緑色マスクとして検出されます。スライダーで検出感度・輪郭線フィルタ・膨張幅を調整し、必要なら **Brush** / **Eraser** / **Rect Fill** で微調整します。
4. **Decensor** をクリックすると復元が実行され、`decensor_output` に自動保存されると同時にウィンドウ内で Before/After を確認できます。マスクを修正して再実行することも可能です。

各ツールやパラメータ・ショートカットの詳細は [Usage (Mask & Decensor)](docs/USAGE.md#iv-mask--decensor-built-in-mask-editor) を参照してください。

独自のイラストでバー用モデルを学習する場合は、[データ準備と学習の手順](docs/TRAINING.ja.md)を参照してください。

# DeepCreamPy
*Plausibly Reconstruct Anime-style Artworks with Deep Neural Networks.*

[![GitHub release](https://img.shields.io/github/release/Deepshift/DeepCreamPy.svg)](https://github.com/Deepshift/DeepCreamPy/releases/latest)
[![GitHub downloads](https://img.shields.io/github/downloads/Deepshift/DeepCreamPy/latest/total.svg)](https://github.com/Deepshift/DeepCreamPy/releases/latest)
[![GitHub downloads](https://img.shields.io/github/downloads/Deepshift/DeepCreamPy/total.svg)](https://github.com/Deepshift/DeepCreamPy/releases)
[![GitHub issues](https://img.shields.io/github/issues/Deepshift/DeepCreamPy.svg)](https://github.com/Deepshift/DeepCreamPy/issues)

A deep learning-based tool to automatically replace parts of artworks with plausible reconstructions.

Before using DeepCreamPy, the user must mark regions in the artwork using green color with an image editing program (e.g., GIMP, Photoshop). DeepCreamPy takes the images with green colored regions as input, and a neural network automatically fills in the highlighted regions.

Legacy Windows 64-bit binaries are available in the [release archive](https://github.com/Deepshift/DeepCreamPy/releases).

The current source setup targets Windows x64 with Python 3.13. See the [uv installation and model conversion instructions](docs/INSTALLATION.md).

Before opening a new issue, please check [closed issues](https://github.com/Deepshift/DeepCreamPy/issues?q=is%3Aissue+is%3Aclosed) and refer to the [table of contents](https://github.com/Deepshift/DeepCreamPy#table-of-contents).

<p align="center">
	<img src="https://github.com/Deepshift/DeepCreamPy/blob/master/readme_images/mermaid_collage.png" width="800">
</p>

## Features
- Reconstructing images of any size
- Reconstruction of ANY shaped censor (e.g. black lines, pink hearts, etc.)
- Decensoring of mosaic censors
- Built-in **Mask & Decensor** editor with drag-and-drop loading, black/white bar auto-detection, manual brush/eraser tools, and live Before/After preview
- Limited support for black-and-white/monochrome images
- Generate multiple variations of reconstructions from the same image

## Limitations
The reconstruction is mainly for anime-style human-like figures that have minor to moderate redactions. If an organ (e.g. arms, legs) is completely deleted, reconstruction will fail.

It does NOT work with:
- Screentones (e.g. printed material)
- Real life photos
- Animated gifs and videos

## Table of Contents
Setup:
* [Running from source with uv](docs/INSTALLATION.md)
* [Running a legacy Windows 64-bit release](docs/INSTALLATION_BINARY.md)

Usage:
* [Tutorial](docs/USAGE.md)
* [Troubleshooting for installing](docs/TROUBLESHOOTING.md)
* [Troubleshooting for poor quality outputs](docs/TROUBLESHOOTING_DECENSORS.md)

Miscellaneous:
* [FAQ](docs/FAQ.md)

## To do
- Improving UI
- Error logging

## Contributions
We welcome contributions as long as they comply with the [GNU Affero General Public License v3.0](LICENSE.md). Be advised of GitHub's [inbound=outbound](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license) rule.

Previously, contributors had to sign a [Contributor License Agreement](https://github.com/deeppomf/contributing/blob/master/sign-cla.md#sign-the-cla) (the "CLA"). This requirement is currently no longer in place.

This project was initially created by **deeppomf** and all credit goes to them. Special thanks to ccppoo, IAmTheRedSpy, 0xb8, deniszh, Smethan, harjitmoe, itsVale, StartleStars, SoftArmpit and everyone else for their contributions!

## License
Source code and official releases/binaries are distributed under the [GNU Affero General Public License v3.0](LICENSE.md).

## Acknowledgements
Example mermaid image by Shurajo & AVALANCHE Game Studio under [CC BY 3.0 License](https://creativecommons.org/licenses/by/3.0/). The example image is modified from the original, which can be found [here](https://opengameart.org/content/mermaid).

Neural network code is modified from Forty-lock's project [PEPSI](https://github.com/Forty-lock/PEPSI), which is the official implementation of the paper [PEPSI : Fast Image Inpainting With Parallel Decoding Network](http://openaccess.thecvf.com/content_CVPR_2019/html/Sagong_PEPSI__Fast_Image_Inpainting_With_Parallel_Decoding_Network_CVPR_2019_paper.html). [PEPSI](https://github.com/Forty-lock/PEPSI) is licensed under the MIT license.

Training data is modified from gwern's project [Danbooru2017: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset](https://www.gwern.net/Danbooru2017) and other sources.

See [ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md) for full license text of these projects.
