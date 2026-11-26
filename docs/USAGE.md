# Usage
## I. Decensoring bar censors

For each image you want to decensor, using image editing software like Photoshop or GIMP to color the areas you want to decensor the green color (0,255,0), which is a very bright green color and be sure that the picture you use has RGB color Mode not Indexed Color Mode.

*I strongly recommend you use the pencil tool and NOT the brush tool.*

*If you aren't using the pencil tool, BE SURE TO TURN OFF ANTI-ALIASING on the tool you are using.*

I personally use the wand selection tool with anti-aliasing turned off to select the censored regions. I then expand the selections slightly to completely cover the censored areas, pick the color (0,255,0), and use the paint bucket tool on the selected regions.

To expand selections in Photoshop, do Selection > Modify > Expand or Contract.

To expand selections in GIMP, do Select > Grow.

Save these images in the PNG format to the "decensor_input" folder. **They MUST be in PNG format.**

### A. Using the binary

Open `DeepCreamPy.exe`, select **Bar censor**, and click **Decensor Your Images**.

### B. Running from source

After [setting up the project with uv](INSTALLATION.md), run bar mode in the CLI.

```powershell
uv run --locked python src/decensor.py
```

Decensored images will be saved to the "decensor_output" folder. Decensoring takes a few minutes per image.

## II. Decensoring mosaic censors

As with decensoring bar censors, perform the same steps of coloring the censored regions green and putting the colored image into the "decensor_input" folder.

In addition, move the original, uncolored images into the "decensor_input_original" folder. Ensure each original image has the same names as their corresponding colored version in the "decensor_input" folder.

For example, if the original image is called "mermaid.jpg," then you want to put this image in the "decensor_input_original" folder and, after you colored the censored regions, name the colored image "mermaid.png" and move it to the "decensor_input" folder.

### A. Using the binary

Open `DeepCreamPy.exe`, select **Mosaic censor**, and click **Decensor Your Images**.

### B. Running from source

Run mosaic mode in the CLI.

```powershell
uv run --locked python src/decensor.py --is_mosaic=true
```

Decensored images will be saved to the "decensor_output" folder. Decensoring takes a few minutes per image.

## III. Decensoring with the user interface

Run `uv run --locked python src/main.py`. Select **Bar censor** or **Mosaic censor**, choose the model and number of variations, then click **Decensor Your Images**. The status bar shows the active device.

## IV. Mask & Decensor (Built-in Mask Editor)

Instead of manually painting `(0, 255, 0)` green masks in an external image editor and placing files into `decensor_input` / `decensor_input_original`, you can click **Mask & Decensor** in the main window to open the built-in mask editor and decensor images directly.

The editor can be opened even while the model is still loading at startup; the **Decensor** action becomes available once loading finishes. Both **Bar** and **Mosaic** modes are supported because the editor keeps the original image and the mask layer separate in memory.

### 1. Loading Images

You can load an image (`PNG`, `JPG`, `JPEG`, `WebP`, or `BMP`) in any of the following ways:

- **Drag & Drop** an image file anywhere onto the Mask & Decensor window.
- Click **Open Image...** (or press `Ctrl+O`).
- Click **Paste** (or press `Ctrl+V`) to paste an image directly from the clipboard.

Dropping or opening another image switches immediately without confirmation dialogs for fast sequential editing.

### 2. Auto-Detecting Black / White Censor Bars

With the **Auto Detect** tool selected (default in **Bar** mode):

1. Drag a rectangle around the censored area. Detected pixels are highlighted as a **pending** preview inside a dashed bounding box.
2. Fine-tune the pending detection in real time using the **Auto Detect (Bar)** controls in the sidebar:
   - **Black / White / Both**: Choose whether to detect dark bars (`Black`, default), bright/white bars (`White`), or both (`Both`).
   - **Black Max**: Upper luminance threshold for black censor detection (`0–160`). Higher values include lighter grays.
   - **White Min**: Lower luminance threshold for white censor detection (`120–255`). Lower values include softer white glows.
   - **Max Chroma**: Maximum RGB channel spread (`0–120`) to reject colorful artwork regions while tolerating slight JPEG compression noise.
   - **Line Filter**: Minimum stroke thickness in pixels (`0–12`) using morphological opening to exclude thin black anime lineart. Set to `0` to disable filtering and detect 1px lines.
   - **Dilation**: Expands detected regions outward (`0–10` px) to cover anti-aliased or compressed edges around censor bars.
3. Commit the pending detection into the mask layer by pressing `Enter`, clicking **Apply (Enter)**, switching tools, drawing the next rectangle, or clicking **Decensor**. Press `Esc` or click **Cancel (Esc)** to discard the pending detection.

When switching **Censor Type** to **Mosaic**, the active tool automatically switches to **Rect Fill** (and switches back to **Auto Detect** when returning to **Bar**).

### 3. Manual Editing Tools & Canvas Navigation

- **Auto Detect**: Drag a rectangle to auto-detect black/white bars inside the selection.
- **Rect Fill**: Left-drag to fill a rectangle with mask; right-drag to erase a rectangle from the mask.
- **Brush**: Paint mask strokes manually. Adjust radius via the **Brush Size** slider or `[` / `]` keys.
- **Eraser**: Erase false positives or unwanted mask pixels manually.
- **Undo / Redo / Clear Mask**: Up to 40 undo steps (`Ctrl+Z` / `Ctrl+Y` or `Ctrl+Shift+Z`). Pressing `Undo` while a pending auto-detection is active cancels the pending detection first.
- **Mask Alpha**: Adjust the transparency of the green mask overlay (`10%–100%`) so you can inspect the underlying artwork while editing.
- **Zoom & Pan**: Scroll the mouse wheel to zoom around the cursor, use `-` / `Fit` (`0`) / `+` in the toolbar, and middle-drag or hold `Space` + left-drag to pan.

### 4. Decensoring, Before/After Preview, and Saving

- **Decensor Settings**: `Type` (`Bar` / `Mosaic`), `Model` (`PEPSI` / `LaMa`), `Use TTA`, and `Variations` (`1` / `2` / `4`) are available directly in the sidebar and stay two-way synced with the main window.
- Click **Decensor** to run in-memory inference using the shared loaded model:
  - Output images are automatically saved to `decensor_output` as lossless PNG files named `<source_name> <0..N>.png` (or `clipboard_<timestamp> <0..N>.png` for pasted images). Re-running on the same image overwrites the previous outputs for that image.
  - Files in `decensor_input` / `decensor_input_original` are not touched, even if **Clean up input directories after decensoring** is checked in the main window.
- Once finished, the canvas switches to **Result** view while keeping your current zoom and pan position:
  - Toggle between **Edit** and **Result** in the top toolbar to inspect the output, tweak the mask, and re-run **Decensor** as many times as needed.
  - Press and hold **Hold to Compare** to temporarily show the original image (Before/After comparison).
  - Use the **Variation** dropdown to switch between generated variations when `Variations > 1`.
- Optional manual export buttons in the sidebar:
  - **Save Result As...**: Save the currently displayed variation to any location as PNG, JPEG, or WebP.
  - **Save Masked PNG...**: Export the image with exact `(0, 255, 0)` mask pixels applied as a PNG file (for reuse in batch mode or external tools).

### 5. Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl+O` | Open image file |
| `Ctrl+V` | Paste image from clipboard |
| `Ctrl+Z` | Undo (or cancel pending auto-detect) |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Enter` | Apply pending auto-detection |
| `Esc` | Cancel pending auto-detection |
| `[` / `]` | Decrease / increase brush size |
| `0` | Fit image to window |
| `Space` + Left-drag / Middle-drag | Pan canvas |
| Mouse wheel | Zoom in / out around cursor |

# Troubleshooting

If you have difficulties getting DeepCreamPy to decensor, go [here](TROUBLESHOOTING.md).
