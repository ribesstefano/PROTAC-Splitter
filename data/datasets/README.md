---
configs:
- config_name: standard
  data_files:
  - split: train
    path: "standard/train.csv"
  - split: test
    path: "standard/test.csv"
- config_name: hardest
  data_files:
  - split: train
    path: "hardest/train.csv"
  - split: test
    path: "hardest/test.csv"
- config_name: e3_unique
  data_files:
  - split: train
    path: "e3_unique/train.csv"
  - split: test
    path: "e3_unique/test.csv"
- config_name: linker_unique
  data_files:
  - split: train
    path: "linker_unique/train.csv"
  - split: test
    path: "linker_unique/test.csv"
- config_name: poi_unique
  data_files:
  - split: train
    path: "poi_unique/train.csv"
  - split: test
    path: "poi_unique/test.csv"
---