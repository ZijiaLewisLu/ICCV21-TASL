# ICCV21-TASL

The official github for our ICCV 2021 paper [Weakly-Supervised Action Segmentation and Alignment via Transcript-Aware Union-of-Subspaces Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Weakly-Supervised_Action_Segmentation_and_Alignment_via_Transcript-Aware_Union-of-Subspaces_Learning_ICCV_2021_paper.pdf).

<!-- The code and dataset will be released soon. -->

!(main_graph)[resources/schematic.png]

## Dataset

Please download data from links below and place them in a `datasets` folder.
- [Breakfast](https://drive.google.com/file/d/1Af3EdYtlxjgBHpvANk60-d6nI6lIkWaf/view?usp=sharing)
- [Hollywood](https://drive.google.com/file/d/13dwHn8JNBM045GzOZoh7KWDxRPUZ2MKL/view?usp=sharing)
- [CrossTask](https://drive.google.com/file/d/1BsPo-hS9fduJYN-ZRZo1azabEjh-rtdT/view?usp=sharing)
## Training

To train model for a dataset, e.g., Breakfast, use `bash breakfast.sh`.

It will automatically generate a `log` folder to save network checkpoints and evaluation results.


## Generate transcript group for *Hierarchical Segmentation* 

Please see `src/utils/cluster_transcript.py`.

## Citation 
Please cite us with
```
@InProceedings{Lu_2021_ICCV,
    author    = {Lu, Zijia and Elhamifar, Ehsan},
    title     = {Weakly-Supervised Action Segmentation and Alignment via Transcript-Aware Union-of-Subspaces Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8085-8095}
}
```
