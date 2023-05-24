pip install waymo-open-dataset-tf-2-4-0==1.4.1 

### Data Preparation
```
# extract image,range_image,calibration (about 2 hours)
python ./process_data/waymo2range.py --process 24
# extract sceneflow and extra info
python ./process_data/range2proposals.py --process 24
# extract point cloud (remove ground)
python ./process_data/waymo2point_vis --process 24
```

```
### Data Format
```
waymo_nsfp
├── segment-xxxx/
│   ├── image/
│   ├── range/
│   ├── PC_ng/
│   ├── sceneflow_extra/
│   ├── calibration.txt 
```