# 独自のイラストでバー用モデルを学習する

DeepCreamPy のバー用生成器をランダム初期化し、欠損のないイラストから学習します。モデルには、指定した範囲に小さなバーを入れた画像を与え、バーで隠れた部分の RGB 値を正解画像に近づけます。実行前にリポジトリ直下で `uv sync --locked` を実行してください。

## 1. 学習対象を矩形で囲む

`dataset/images` に欠損のない PNG、JPEG、WebP を置きます。サブフォルダも使えます。学習と検証に分けるため、矩形を付けた画像が最低2枚必要です。画像の短辺は8ピクセル以上にしてください。

```powershell
uv run --locked python src/annotate_people.py --images dataset/images --annotations dataset/people.csv
```

画像上で人物をドラッグして囲みます。1枚に複数の矩形を追加できます。編集するには右側の一覧から矩形を選び、描き直すか削除します。変更は CSV に自動保存されます。同じコマンドで再度起動すると、最初の矩形が未登録の画像から開きます。

特定部位の復元に絞る場合は、手や顔など、対象部位を囲む矩形だけを登録します。バーは矩形の内側に生成され、モデルには矩形の外側も含めたクロップが渡ります。対象部位の輪郭に沿って矩形を小さく取ると、バーが背景だけを隠す例を減らせます。ポーズや向きの異なる例も含めてください。

CSV を手で作る場合は、次の形式で UTF-8 で保存します。パスは `--images` からの相対パス、座標は元画像の左上を `(0, 0)` としたピクセル値です。右端と下端の座標は矩形に含めません。

```csv
image,x_min,y_min,x_max,y_max
sample.png,65,40,250,390
sample.png,300,48,482,408
chapter-1/scene.jpg,82,100,449,640
```

## 2. 学習する

`--output` に学習結果専用の新しいフォルダを指定します。

```powershell
uv run --locked python src/train_bar.py --images dataset/images --annotations dataset/people.csv --output models/bar/runs/my-bar --epochs 20
```

CUDA とバッチサイズ1が既定です。エポックごとに各矩形の内側で横長か縦長のバーを抽選します。モデル入力の短辺4〜12 px、長辺24〜64 px程度が目安です。小さな矩形や画像では、収まる大きさに調整します。画像はアプリと共通の処理で周囲ごと切り出し、256 × 256 に収めます。損失には隠れたピクセルの平均絶対誤差（L1）を使います。

矩形を登録した画像の約10%を検証に使います。同じ画像の複数の矩形は同じ側にまとめ、検証画像と検証用のバーはエポック間で固定します。

| ファイル | 用途 |
| --- | --- |
| `inpaint.pt` | 検証 L1 が最も小さかったエポックの、アプリ用の重み |
| `last.pt` | 最新の完了エポックから再開するための重みとオプティマイザーの状態 |
| `preview-0001.png` など | 固定した検証例の欠損入力、復元結果、正解画像を左から順に表示 |

20エポックから40エポックまで続ける場合は、同じ `--images`、`--annotations`、`--output` を指定し、`--resume` を付けます。`--epochs` は再開後に追加する数ではなく、累計のエポック数です。`--seed` を変更していた場合は、再開時も同じ値を指定します。

```powershell
uv run --locked python src/train_bar.py --images dataset/images --annotations dataset/people.csv --output models/bar/runs/my-bar --epochs 40 --resume
```

## 3. アプリの CLI で使う

復元したい範囲を緑で塗った PNG を `decensor_input` に置き、学習した重みのパスを渡します。

```powershell
uv run --locked python src/decensor.py --bar-weights models/bar/runs/my-bar/inpaint.pt
```

復元した画像は `decensor_output` に保存されます。
