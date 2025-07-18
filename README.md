# GLPN-LLM

Source code (PyTorch) and dataset of the paper "[Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](https://arxiv.org/abs/2506.00488)", which is accepted by The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025).


## Dataset
You can download the dataset from [here](https://drive.google.com/file/d/1gPX-tAC1Vo6C8j8PV9IbAk8hbDhd1XMG/view?usp=drive_link).

Unzip the dataset and put it in the `script/dataset` folder. We have three datasets: weibo, twitter and pheme. The dataset is in the form of csv files. 

The file structure is as follows:
```
script/dataset/
    weibo/
    twitter/
    pheme/
```


## Requirements
You can install the requirements by running the following command:
```bash
pip install -r requirements.txt
```
Note: It's recommended to install the CLIP package directly from the [official GitHub repository](https://github.com/openai/CLIP.git).

## Running
You can run the code by running the following command:
```bash
sh run.sh
```
Note: you can use the psesudo labels generated by GPT-4o to train the model, `dataset/{args.dataset_name}/{args.dataset_name}_analysis_results.csv` is the file that stores the psesudo labels.


## Cite
If you use GLPN-LLM in a scientific publication, we would appreciate citations to the following paper:

```
@article{hu2025synergizing,
  title={Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection},
  author={Hu, Shuguo and Hu, Jun and Zhang, Huaiwen},
  journal={arXiv preprint arXiv:2506.00488},
  year={2025}
}
```

License: GPLv3

Copyright (c) 2024-2025 IMU, China & NUS, Singapore.
