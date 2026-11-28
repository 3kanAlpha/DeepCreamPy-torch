# DeepCreamPy のインペインティングモデル

[English](./MODELS.md) | 日本語

DeepCreamPy は、RGB 画像内のマスク領域を補完するために、PEPSI ベースの畳み込みエンコーダ・デコーダを採用しています。ボトルネック部に配置された Contextual Attention モジュールが、同一クロップ内から特徴パッチを照合・合成することで、欠損部分の自然な生成を助けます。PyTorch 移植版では、この Attention を用いたインペインティング経路を 1 回の順伝播（フォワードパス）で実行します。[PyTorch モデル](../src/model.py) · [PEPSI 論文 (Sagong et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Sagong_PEPSI__Fast_Image_Inpainting_With_Parallel_Decoding_Network_CVPR_2019_paper.html)

## バーおよびモザイク用チェックポイント

どちらのモードでも同じ `InpaintModel` に異なる学習済み重みを読み込みます。マスク検出では指定色と完全一致する画素を起点に、連結する最大 2 ピクセル幅のアンチエイリアス縁（フリンジ）を自動で取り込みます。アプリケーションは連結したマーク領域を周辺ごと切り出し、縦横比を保って 256 × 256 の入力に収めます。RGB のリサイズには、縮小時は `BOX`、拡大時は未マスク画素のみを用いた正規化 `BILINEAR` 補間を使い、マスク色の滲みを防ぎつつ線画の劣化を抑えます。マスクを縮小するときは、元のマークを含む画素を補完対象にします。余白を埋めるときや画像端のマークに余白を足すときは、画像とマスクの両方を反射させます。エンコーダには `[-1, 1]` に正規化した RGB と、3 チャンネルに複製したマスクを渡します。マスク値は 1 が保持、0 が補完を表します。[クロップの前処理](../src/libs/utils.py) · [モデルの読み込み](../src/model.py)

| モード | チェックポイント | マスク内の RGB 値 |
| --- | --- | --- |
| バー (Bar) | `models/bar/inpaint.pt` | 正規化前に黒で塗りつぶされるため、正規化後は `-1` になります。 |
| モザイク (Mosaic) | `models/mosaic/inpaint.pt` | モザイクが残ったままの対応する元画像から取得します。マスクは色付けされたコピー画像から生成されます。 |

## ネットワークアーキテクチャ

生成器（ジェネレータ）は 20 個の畳み込み層で構成され、バイアスを含めて合計 3,656,947 個のパラメータを持ちます。下表の出力形状は、1 つの入力クロップに対する「チャンネル数 × 高さ × 幅」で表しています。[レイヤー定義](../src/model.py)

| ステージ | 処理内容 | 出力形状 |
| --- | --- | --- |
| エンコーダ | 5 × 5 の畳み込み 1 層に続き、3 × 3 の畳み込みを 5 層適用します。各層の出力チャンネル数は順に 32、64、64、128、128、256 で、2・4・6 層目ではストライド 2 を使用します。 | 256 × 32 × 32 |
| 拡張畳み込みエンコーダ (Dilated encoder) | 拡張率（dilation rate）2、4、8、16 の 3 × 3 畳み込みを 4 層適用し、解像度を維持したまま受容野を広げます。 | 256 × 32 × 32 |
| Contextual Attention | マスク内の特徴を再構成し、得られた特徴マップをエンコーダ出力とチャンネル方向に結合します。その後、1 × 1 の畳み込みで 512 チャンネルから 256 チャンネルへ射影します。 | 256 × 32 × 32 |
| デコーダ | 出力チャンネル数がペアごとに 128、64、32、16 となる 3 × 3 畳み込みを 4 ペア適用します。最初の 3 ペアの末尾では、それぞれ最近傍補間による 2 倍アップサンプリングを行います。 | 16 × 256 × 256 |
| RGB 出力 | ゼロパディング付きの 3 × 3 畳み込みを 1 層適用し、値を `[-1, 1]` の範囲にクリッピングします。 | 3 × 256 × 256 |

エンコーダおよびデコーダの中間層における畳み込みでは、反射パディング（reflection padding）と活性化関数 ELU を使用しています。コンテキスト射影用の 1 × 1 畳み込みにも ELU が使われています。[順伝播の実装](../src/model.py)

### Contextual Attention と画像合成

Attention モジュールは、まずマスクの 1 チャンネル目を最近傍補間で 32 × 32 にダウンサンプリングします。次に、コピーしたエンコーダ特徴マップのマスク領域をゼロで埋めたうえで、3 × 3 の候補パッチを抽出します。これらの候補パッチとエンコーダ特徴マップ上のクエリパッチを二乗ユークリッド距離で比較します。各クエリについて候補間の距離を標準化し、`-50 * tanh(standardized_distance)` を適用したあと、候補全体に対してソフトマックスを計算します。最後に、重み付けしたパッチを特徴マップへと折り畳み直し（fold）、1 × 1 射影の前にマスク領域の特徴を補完するために使用します。[Contextual Attention の実装](../src/model.py)

順伝播の最後では、デコードされた RGB 画像と元のクロップ画像を `prediction * (1 - mask) + source * mask` で合成します。推論時は既定で 4 ビューの推論時拡張（元画像・左右反転および `(2, 2)` ピクセルの反射シフト対）を実行して生デコード結果を位置合わせ平均します（CLI の `--no-tta` または GUI のチェックボックスで無効化可能）。その後、余白を取り除いた結果を浮動小数点のままバイキュービック補間でクロップの元のサイズに戻し、境界外周の残差をマスク内部へ滑らかに伝播するシームレス補正と、周辺がモノクロの場合の不要な色づき抑制を行ってから 8 ビットに変換します。予測結果はマークされた画素にだけ貼り付けます。幅1ピクセルの欠損は、両側の画素の色が近い場合にその平均色で補正し、周囲の線や塗りにつなげます。補完した画素は、入力画像とその周囲から生成した推定値です。[モデル内の合成処理](../src/model.py) · [画像合成処理](../src/decensor.py)

## 学習グラフと重みの変換

オリジナルの TensorFlow 計算グラフでは、共有エンコーダ 1 つに対し、共有デコーダを 2 つの経路（パス）で通す構成になっています。1 つ目の粗い再構成パス（coarse pass）はエンコーダの特徴をそのままデコードし、補助的な L1 再構成損失を算出します。2 つ目のインペインティングパス（inpainting pass）は Contextual Attention で精緻化した特徴をデコードし、L1 再構成損失に加えて Region Ensemble Discriminator (RED) による敵対的損失（adversarial loss）を使用します。[DeepCreamPy 2.2.0 Beta の TensorFlow グラフ](https://github.com/Deepshift/DeepCreamPy/blob/2.2.0-beta/model.py) · [PEPSI 論文 Section 3](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sagong_PEPSI__Fast_Image_Inpainting_With_Parallel_Decoding_Network_CVPR_2019_paper.pdf)

変換スクリプトは、学習済み TensorFlow モデルからエンコーダ（`G_en`）、コンテキスト射影（`CB1/ML`）、デコーダ（`G_de`）の重みとバイアスを読み込みます。そして、畳み込みカーネルの軸配列を TensorFlow 形式（高さ × 幅 × 入力チャンネル × 出力チャンネル）から PyTorch 形式（出力チャンネル × 入力チャンネル × 高さ × 幅）へと並べ替え、モードごとに 1 つの state dict として保存します。[変換スクリプト](../src/convert_models.py) · [モデルのセットアップ](INSTALLATION.md#prepare-the-models)

## LaMa モデル（バーモード専用）

バーモードでは、PEPSI に加えて Fast Fourier Convolutions (FFC) を用いた Pure PyTorch 実装の **LaMa (`FFCResNetGenerator`)** を選択できます。[LaMa 実装](../src/lama.py) · [LaMa 論文 (Suvorov et al., WACV 2022)](https://arxiv.org/abs/2109.07161)

* **チェックポイント**: 既定では `models/lama/inpaint.pt` を読み込みます（CLI の `--bar-weights` で別ファイルも指定可）。Big-LaMa（18 ブロック、約 51M パラメータ）および標準 LaMa（9 ブロック、約 27M パラメータ）のどちらの構成も `state_dict` から自動判別します。
* **前処理**: クロップの長辺が 512 px を超える場合のみ縦横比を保って `LANCZOS` 縮小し（512 px 以下は拡大しない）、画像とマスクを反射パディングで 8 の倍数に揃えます。入力は `[0, 1]` に正規化した RGB（マスク内は 0 埋め）と、`1 = 補完対象, 0 = 保持` の 1 チャンネルマスクを結合した 4 チャンネルです。[LaMa クロップ処理](../src/libs/utils.py) · [LaMa 推論ラッパー](../src/model.py)
* **重みの変換**: 既存の学習済みチェックポイント（TorchScript `.pt`、PyTorch Lightning `.ckpt`、`.safetensors`）は変換スクリプトで `models/lama/inpaint.pt` に変換できます。[LaMa 変換スクリプト](../src/convert_lama.py)

```powershell
uv run --locked python src/convert_lama.py path/to/big-lama.pt
```

`.safetensors` を変換する場合は `--with safetensors` を付けて実行します。CLI での実行は `--model lama` を指定してください。GUI では「Model」グループから `LaMa (Bar only)` を選択できます（Mosaic モード選択時は無効化されます）。

## Anima モデル + 4ch インペイント ControlNet-LLLite（バーモード・GPU 専用）

CUDA 対応 NVIDIA GPU 環境でオプション依存グループ `anima` を導入している場合、バーモードで **Anima（`Anima-Base` + 4ch インペイント用 ControlNet-LLLite）** を選択できます。[Anima 推論エンジン](../src/anima.py) · [ControlNet-LLLite アダプタ](../src/anima_lllite.py)

* **オプション依存の導入**: `uv sync --locked --group anima` で導入します（PEPSI / LaMa のみを使う場合は不要です）。
* **ウェイトファイルの配置（完全オフライン）**: ComfyUI 配布と同じ単体 `.safetensors` ファイルを `models/anima/` 以下の 4 フォルダへ配置します（Anima の非商用ライセンスへの配慮のため、配布物には同梱されません）。
  * `models/anima/diffusion/` — `Anima-Base` 本体（例: `anima-base-v1.0.safetensors`）
  * `models/anima/text_encoder/` — Qwen3-0.6B テキストエンコーダ（例: `qwen_3_06b_base.safetensors`）
  * `models/anima/vae/` — Qwen-Image VAE（例: `qwen_image_vae.safetensors`）
  * `models/anima/controlnet/` — 4ch インペイント用 ControlNet-LLLite（例: `anima-lllite-inpainting-v2.safetensors`）
  * `diffusion/` や `controlnet/` に複数ファイルを配置した場合は、詳細設定内のドロップダウンから切り替えられます。
* **トークナイザ**: 軽量な Qwen3 / T5 トークナイザ定義を `assets/anima_tokenizers/` に同梱しているため、推論時にネット接続は不要です。
* **8GB VRAM 対策と FP8 ストレージ量子化**: `bfloat16`（非対応 GPU は `float16`）を基本とし、コンポーネント順次オフロード・CFG 逐次実行（batch=1）・VAE タイリング／スライシングを行います。VRAM 10.5 GiB 以下（8GB クラス等）の環境では **FP8 ストレージ量子化（`float8_e4m3fn`、`diffusers` の layerwise casting）** が既定で有効になり、DiT を約 2.1GB に圧縮して GPU 常駐のまま高速に推論できます（GUI の「Advanced Settings」や CLI の `--anima-fp8` / `--no-anima-fp8` でオンオフ可能）。FP8 を無効にした 8GB クラス環境や 6.5 GiB 以下の環境ではブロック単位オフロードが自動で有効になります。
* **インペイント・クロップ統合・Variations**: 近接するマスク領域を広めの余白付きクロップへ自動統合（中サイズ以下の画像や広範囲欠損は画像全体を一括推論）し、16 の倍数へアライメントします。純ノイズ開始（Denoise 1.0）＋ 4ch LLLite 条件付け ＋ ステップ毎の既知領域 Latent 維持で生成し、マスク内画素のみ貼り戻して境界・モノクロ補正を適用します。Variations は反転ではなくシード違いで生成します。
* **補助プロンプトと実行方法**: 絵柄維持のため Positive は空欄デフォルト（必要に応じて部位や `@artist` 等を補助入力）、Negative は編集可能な初期値（`censored, bar censor, mosaic censoring`・消去可）となっています。GUI の **Model → Anima** または CLI（`--model anima`）から利用できます。
