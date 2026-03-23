# FilaSeg

Tools for vectorizing and reconnecting filamentous structures in microscopy images.

The **string art vectorization** module is novel work — it decomposes binary filament masks into oriented line/curve primitives using a tile-wise greedy reconstruction, as an alternative to CNN-based instance separation. The **reconnection** module is a significantly extended Python reimplementation of the terminus pairing concept introduced in the papers below, replacing the original MATLAB implementation with smooth Bézier bridging and geometric validation.

## Related work

* Liu et al., *Densely Connected Stacked U-Network for Filament Segmentation in Microscopy Images*, ECCV Workshops, 2018. [link](http://openaccess.thecvf.com/content_eccv_2018_workshops/w33/html/Liu_Densely_Connected_Stacked_U-network_for_Filament_Segmentation_in_Microscopy_Images_ECCVW_2018_paper.html)
* Liu et al., *Intersection To Overpass: Instance Segmentation on Filamentous Structures with An Orientation-Aware Neural Network and Terminus Pairing Algorithm*, CVPR Bioimaging Workshop, 2019. [link](http://openaccess.thecvf.com/content_CVPRW_2019/paper/BIC/Liu_Intersection_to_Overpass_Instance_Segmentation_on_Filamentous_Structures_With_an_CVPRW_2019_paper.pdf)

Original pipeline: [VimsLab/filaments_quantification](https://github.com/VimsLab/filaments_quantification)

## Environment

```bash
pip install -r requirements.txt
# or use the provided conda environment:
conda env create -f mtquant.yml
```

## Modules

### `stringart/` — Vectorization

Converts binary filament masks into vector representations using a tile-wise greedy reconstruction approach.

| File | Description |
|------|-------------|
| `stringart_tiles.py` | Primary script — line-based, with automatic width estimation and parameter scaling |
| `stringart_tiles_curve.py` | Curve variant — pure quadratic Bézier primitives with multi-stage curvature progression |

### `reconnect/` — Filament reconnection

Reconnects broken filament fragments at gaps and intersections using smooth Bézier bridging with geometric validation.

| File | Description |
|------|-------------|
| `reconnect_run.py` | Main entry point — select algorithm version via `--version` |
| `reconnect_utils_v5.py` | Baseline reconnect implementation with smooth bridging and geometric validation |
| `reconnect_utils_v6.py` | Reconnect implementation with improved tip enumeration and stability handling |
| `reconnect_config.yaml` | Shared configuration for both versions |
| `reconnect_interactive.py` | Interactive slider-based parameter tuning tool |

**Usage:**

```bash
# Run with default (v6)
python reconnect/reconnect_run.py --config reconnect/reconnect_config.yaml

# Run with v5
python reconnect/reconnect_run.py --version v5

# Override input/output paths
python reconnect/reconnect_run.py --version v6 --input ./my_input --output ./my_output

# Compare extracted stats against ground truth JSON
python reconnect/reconnect_run.py --version v6 --compare --no_show
```
