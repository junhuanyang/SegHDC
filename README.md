# On-Device Unsupervised Image Segmentation

## Implementation of SegHDC

Repo for "[On-Device Unsupervised Image Segmentation](https://arxiv.org/abs/2303.12753)", accepted by [DAC 2023](https://www.dac.com/).

## The Framework
![Framework](./imgs/framework.png)

## Dependencies

### Prerequisites
- Python 3.9.7

### Install the necessary packages.
```bash
pip install -r requirements.txt
```

### Run:
```bash
python process_test_PDMXORInit3Chans.py 
```
to test the images (in "test" directory) and get the segmentation mask (stored in the "bestMask" directory), and IoU score. 

### Configuration
Modify the hyperparameters and configurations in 
```
arguparse.py
```

### Citation
If you find the code and our idea helpful in your research or work, please cite the following paper.

```
@inproceedings{yang2023device,
  title={On-device unsupervised image segmentation},
  author={Yang, Junhuan and Sheng, Yi and Zhang, Yuzhou and Jiang, Weiwen and Yang, Lei},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```