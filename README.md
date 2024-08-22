# Enhancing Object Detection in Images through Hyperbolic Space Representations

Hyperbolic representation has demonstrated promising results in classification scenarios by capturing complex hierarchical relation- ships in classification tasks. This study aims to investigate the integration of hyperbolic space representation in object detection scenarios to improve precision and robustness. Traditional object detection methods typically rely on the Euclidean space, which may limit the effective representation of diverse object classes. Since it can represent hierarchical data, hyperbolic space presents a com- pelling alternative. We utilize hyperbolic embeddings and integrate these transformations into the architecture of the Faster R-CNN model. Leveraging a diverse dataset of MS COCO, we thoroughly evaluate, compare, and analyze the models. The Faster R-CNN model undergoes modifications to incorporate hyperbolic represen- tation in the RoI head’s classification branch, and a specialized loss function is employed. Additionally, prototype learning is utilized to represent ideal classes in hyperbolic space.
While the experimental models did not outperform the baseline Faster R-CNN architecture, the results highlight significant find- ings regarding the application of hyperbolic representation. The research offers valuable insights into the potential of hyperbolic rep- resentation by iteratively utilizing various hyperbolic dimensions for future researchers in the field.

## Results

### Mean Average Precision

| **Model / Metric** | **mAP** | **mAP@50** | **mAP@75** | **mAP-s** | **mAP-m** | **mAP-l** |
|--------------------|---------|------------|------------|-----------|-----------|-----------|
| **Baseline**       | **0.286** | **0.469**  | **0.307**  | **0.164** | **0.312** | **0.370** |
| **20D**            | **0.057** * | **0.085** * | **0.063** * | **0.029** * | **0.053** * | **0.075** * |
| **40D**            | 0.040   | 0.060      | 0.043      | 0.018     | 0.043     | 0.055     |
| **81D**            | 0.022   | 0.032      | 0.024      | 0.012     | 0.022     | 0.030     |
| **100D**           | 0.014   | 0.021      | 0.015      | 0.007     | 0.015     | 0.017     |
| **20D - W**        | 0.023   | 0.038      | 0.024      | 0.013     | 0.022     | 0.035     |
| **40D - W**        | 0.036   | 0.065      | 0.035      | 0.020     | 0.031     | 0.050     |
| **81D - W**        | 0.032   | 0.057      | 0.033      | 0.016     | 0.028     | 0.048     |
| **100D - W**       | 0.034   | 0.061      | 0.033      | 0.018     | 0.027     | 0.049     |

### Average Recall

| **Model / Metric** | **AR@100** | **AR@300** | **AR-s** | **AR-m** | **AR-l** |
|--------------------|------------|------------|----------|----------|----------|
| **Baseline**       | **0.465**  | **0.465**  | **0.268** | **0.503** | **0.596** |
| **20D**            | 0.122      | 0.122      | 0.044    | 0.106    | 0.178    |
| **40D**            | 0.086      | 0.086      | 0.028    | 0.080    | 0.130    |
| **81D**            | 0.054      | 0.054      | 0.017    | 0.044    | 0.085    |
| **100D**           | 0.028      | 0.028      | 0.009    | 0.024    | 0.044    |
| **20D - W**        | 0.212      | 0.212      | 0.092    | 0.212    | 0.306    |
| **40D - W**        | **0.246** *| **0.246** *| 0.128    | **0.235** *| **0.319** *|
| **81D - W**        | 0.230      | 0.230      | 0.118    | 0.216    | 0.287    |
| **100D - W**       | 0.244      | 0.244      | **0.134** *| 0.229    | 0.310    |

We conducted eight epxerimental setups. We utilized 20, 40, 81, and 100-dimensional hyperbolic space, both with regular, the class weighted Penalized Busemann Loss. The `W` represents a class weighted loss used during the training.

## Instructions
- run the `mmdet/utils/prototype_learning.py` script to generate the prototypes. (slurm script is also available in the `tools` repository.)
- As a second step, any of the configurations can be run to train the Hyperbolic Faster R-CNN model. To modify the setup, a new folder should be created inside the `configs/faster-rcnn/hyperbolic_faster_rcnn/` repository, to keep the experimental setups separated.

## NOTICE
For exact changes, please refer to the NOTICE file as per the requirements of the Apache 2.0 License of the orignal project, that can be found in the LICENSE file.


## Reference
```
@mastersthesis{makacs_2024_hyperbolic_object_detection,
  title        = {Enhancing Object Detection in Images through Hyperbolic Space Representations},
  author       = {Ákos Makács},
  year         = {2024},
  month        = {June},
  address     = {1012 WP Amsterdam},
  school       = {University of Amsterdam},
  note         = {Available at: {\url{TODO}}, GitHub repository: \url{https://github.com/mkcsakos/mmdetection}},
  type         = {Master's thesis},
}
```