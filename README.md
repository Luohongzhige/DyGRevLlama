# README
## how to run
In order to avoid excessive memory usage, we recommend that you modify the config file and run it in the following order.The config file is `./data/config/cloth.json`
### download dataset
from https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/ to download dataset and move the file to `./data/cloth/`
### first
change `skip.part4, skip.part5` to `False` and run `python ./work/all.py`
### second
change `skip.part1, skip.part2, skip.part3` to `False` , `skip.part4` to `True` and run `python ./work/all.py`
### third
change `skip.part4` to `False` , `skip.part5` to `True` and run `python ./work/all.py`
