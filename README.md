# README
## how to run
In order to avoid excessive memory usage, we recommend that you modify the config file and run it in the following order.The config file is `./data/config/cloth.json`
### download dataset
from https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/ to download dataset and move the file to `./data/cloth/`
### download Llama
from `transformers` to download llama and replace the `llama_path` in `./work/all.py` from `YOUR_LLAMA_PATH` to the actual location of your Llama model
### first pass
change `skip.part4, skip.part5` to `False` and run `python ./work/all.py`
### second pass
change `skip.part1, skip.part2, skip.part3` to `False` , `skip.part4` to `True` and run `python ./work/all.py`
### third pass
change `skip.part4` to `False` , `skip.part5` to `True` and run `python ./work/all.py`
