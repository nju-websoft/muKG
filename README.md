<div align=center>
<img src="https://github.com/nju-websoft/muKG/blob/main/figs/logo.png" width="375" style="zoom:15%;" />
</div>


> **μKG** is an open-source Python library for representation learning over knowledge graphs. μKG supports joint representation learning over multi-source knowledge graphs (and also a single knowledge graph), multiple deep learning libraries (PyTorch and TF2), multiple embedding tasks (link prediction, entity alignment, entity typing, and multi-source link prediction), and multiple parallel computing modes (multi-process and multi-GPU computing).


## Table of contents

1. [Introduction of μKG 📃](#introduction-of-mukg-)
   1. [Overview](#overview)
   2. [Package Description](#package-description)
2. [Getting Started 🚀](#getting-started-)
   1. [Dependencies](#dependencies)
   2. [Installation](#installation-)
   3. [Usage](#usage-)
3. [Models hub 🏠](#models-hub-)
   1. [KGE models](#kge-models)
   2. [EA models](#ea-models)
   3. [ET models](#et-models)
4. [Datasets hub 🏠](#datasets-hub-)
   1. [KGE datasets](#kge-datasets)
   2. [EA datasets](#ea-datasets)
   3. [ET datasets](#et-datasets)
5. [Utils 📂](#utils-)
   1. [Sampler](#sampler)
   2. [Evaluator](#evaluator)
   3. [ET datasets](#et-datasets)
   4. [Multi-GPU and multi-processing computation](#multi-gpu-and-multi-processing-computation)
6. [Running Experiments 🔬](#running-experiments-)
7. [License](#license)
8. [Citation](#citation)

## Introduction of μKG 📃

### Overview 

We use  [Python](https://www.python.org/) ,  [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) to develop the basic framework of **μKG**.  And using [RAY](https://www.ray.io/) for distributed training. The software architecture is illustrated in the following Figure. 

![image-20220507103409697](https://github.com/nju-websoft/muKG/blob/main/figs/system.png)


Compared with other existing KG systems, μKG has the following competitive features.

👍**Comprehensive.** μKG is a full-featured Python library for representation learning over a single KG or multi-source KGs. It is compatible with the two widely-used deep learning libraries [PyTorch](https://pytorch.org/) and [TensorFlow 2](https://www.tensorflow.org/), and can therefore be easily integrated into downstream applications. It integrates a variety of KG embedding models and supports four KG tasks including link prediction, entity alignment, entity typing, and multi-source link prediction.

⚡**Fast and scalable.** μKG provides advanced implementations of KG embedding techniques with the support of multi-process and multi-GPU parallel computing, making it fast and scalable to large KGs.

🤳**Easy-to-use.** μKG provides simplified pipelines of KG embedding tasks for easy use. Users can interact with μKG with both method APIs and the command line. It also has high-quality documentation.

😀**Continuously updated.** Our team will keep up-to-date on new related techniques and integrate new (multi-source) KG embedding models, tasks, and datasets into μKG. We will also keep improving existing implementations.

  

### 	Package Description

```
μKG/
├── src/
│   ├── py/: a Python-based toolkit used for the upper layer of μKG
		|── data/: a collection of datasets used for knowledge graph reasoning
		|── args/: json files used for configuring hyperparameters of training process
		|── evaluation/: package of the implementations for supported downstream tasks
		|── load/: toolkit used for data loading and processing
		|── base/: package of the implementations for different initializers, losses and optimizers
		|── util/: package of the implementations for checking virtual environment
│   ├── tf/: package of the implementations for KGE models, EA models and ET models in TensorFlow 2
│   ├── torch/: package of the implementations for KGE models, EA models and ET models in PyTorch
```



## Getting Started 🚀

### Dependencies![python3](https://img.shields.io/badge/Python3-green.svg?style=flat-square)

μKG supports PyTorch and TensorFlow 2 deep learning libraries, users can choose one of the following two dependencies according to their preferences.

* Torch 1.10.2  |  Tensorflow 2.x 
* Ray 1.12.0    
* Scipy
* Numpy
* Igraph 
* Pandas
* Scikit-learn
* Gensim
* Tqdm


### Installation 🔧

We suggest you create a new conda environment firstly.  We provide two installation instructions for tensorflow-gpu (tested on 2.3.0) and pytorch (tested on 1.10.2). Note that there is a difference between the Ray 1.10.0 and Ray 1.12.0 in batch generation. The Ray 1.12.0 is used as an example.

```bash
# command for Tensorflow
conda create -n muKG python=3.8
conda activate muKG
conda install tensorflow-gpu==2.3.0
conda install -c conda-forge python-igraph
pip install -U ray==1.12.0
```

To install PyTorch, you must install [Anaconda](https://www.anaconda.com/) and follow the instructions on the PyTorch website. For example, if you’re using CUDA version 11.3, use the following command:

```bash
# command for PyTorch
conda create -n muKG python=3.8
conda activate muKG
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge python-igraph
pip install -U ray==1.12.0
```

The latest code can be installed by the following instructions:

```bash
git clone https://github.com/nju-websoft/muKG.git muKG
cd muKG
pip install -e .
```

### Usage 📝

Currently, there are two ways to do your job. Here we provide tutorials of using command line as well as editing file to configure your model. The following is an example about how to use μKG in Python. You can choose different tasks, select the specific model and change the mode (training or evaluation) here. The hyperparameter files are stored in the  subfolder `args`. It maintains compelete details for training process.

```python
model_name = 'model name'
kg_task = 'selected KG task'
if kg_task == 'ea':
	args = load_args("hyperparameter file folder of entity alignment task")
elif kg_task == 'lp':
	args = load_args("hyperparameter file folder of link prediction task")
else:
	args = load_args("hyperparameter file folder of entity typing task")
kgs = read_kgs_from_folder()
if kg_task == 'ea':
	model = ea_models(args, kgs)
elif kg_task == 'lp':
	model = kge_models(args, kgs)
else:
	model = et_models(args, kgs)
model.get_model('model name')
model.run()
model.test()
```

To run a model on a dataset with the following command line. We show an example of training TransE on FB15K here. The hyperparameters will default to the corresponding json file in the `args_kge` folder.

```bash
# -t:lp, ea, et -m: selected model name -o train and valid -d selected dataset
python main_args.py -t lp -m transe -o train -d data/FB15K
```




## Models hub 🏠

μKG has implemented 26 KG models. The citation for each models corresponds to either the paper describing the model. According to different knowledge graph downstream tasks, we divided the models into three categories. It is available for you to add your own models under one of the three folders.

### KGE models

| Name     | Citation                                                                                                                |
| -------- |-------------------------------------------------------------------------------------------------------------------------|
| TransE   | [Bordes *et al.*, 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| TransR   | [Lin *et al.*, 2015](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/)                           |
| TransD   | [Ji *et al.*, 2015](http://www.aclweb.org/anthology/P15-1067)                                                           |
| TransH   | [Wang *et al.*, 2014](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)                          |
| TuckER   | [Balažević *et al.*, 2019](https://arxiv.org/abs/1901.09590)                                                            |
| RotatE   | [Sun *et al.*, 2019](https://arxiv.org/abs/1902.10197v1)                                                                |
| SimplE   | [Kazemi *et al.*, 2018](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs)     |
| RESCAL   | [Nickel *et al.*, 2011](http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf)                                      |
| ComplEx  | [Trouillon *et al.*, 2016](https://arxiv.org/abs/1606.06357)                                                            |
| Analogy  | [Liu *et al.*, 2017](https://arxiv.org/abs/1705.02426)                                                                  |
| DistMult | [Yang *et al.*, 2014](https://arxiv.org/abs/1412.6575)                                                                  |
| HolE     | [Nickel *et al.*, 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)                      |
| ConvE    | [Dettmers *et al.*, 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366)                              |

### EA models

| Name      | Citation                                                                                                    |
| --------- |-------------------------------------------------------------------------------------------------------------|
| MTransE   | [Chen *et al.*, 2017](https://www.ijcai.org/proceedings/2017/0209.pdf)                                      |
| IPTransE  | [Zhu *et al.*, 2017 ](https://www.ijcai.org/proceedings/2017/0595.pdf)                                      |
| BootEA    | [Sun *et al.*, 2018](https://www.ijcai.org/proceedings/2018/0611.pdf)                                       |
| JAPE      | [Sun *et al.*, 2017](https://link.springer.com/chapter/10.1007/978-3-319-68288-4_37)                        |
| IMUSE     | [He *et al.*, 2019](https://link.springer.com/content/pdf/10.1007%2F978-3-030-18576-3_22.pdf)               |
| RDGCN     | [Wu *et al.*, 2019](https://www.ijcai.org/proceedings/2019/0733.pdf)                                        |
| AttrE     | [Trisedya *et al.*, 2019](https://people.eng.unimelb.edu.au/jianzhongq/papers/AAAI2019_EntityAlignment.pdf) |
| SEA       | [Pei *et al.*, 2019](https://dl.acm.org/citation.cfm?id=3313646)                                            |
| GCN-Align | [Wang *et al.*, 2018](https://www.aclweb.org/anthology/D18-1032)                                            |
| RSN4EA    | [Guo *et al.*, 2019](http://proceedings.mlr.press/v97/guo19c/guo19c.pdf)                                    |

### ET models

| Name   | Citation                                                     |
| ------ | ------------------------------------------------------------ |
| TransE | [Bordes *et al.*, 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| RESCAL | [Nickel *et al.*, 2011](http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf) |
| HolE   | [Nickel *et al.*, 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828) |

## Datasets hub 🏠

μKG has bulit in 16 KG datasets for different downstream tasks. Here we list the number of entities, relations, train triples, valid triples and test triples for these datasets. You can prepare your own datasets in the Datasets hub. Firstly, you should create a subfolder `dataset name` in the `data` folder, then put your train.txt, valid.txt and test.txt files in this folder. The data should be in the triple format.

### KGE datasets
| Datasets Name | Entities | Relations | Train   | Valid | Test    | Citation                                                     |
| ------------- | -------- | --------- | ------- | ----- | ------- | ------------------------------------------------------------ |
| FB15K         | 14951    | 1345      | 483142  | 50000 | 59071   | [Bordes *et al*., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| FB15K237      | 14541    | 237       | 272115  | 17535 | 20466   | [Bordes *et al*., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| WN18RR        | 40943    | 11        | 86835   | 3034  | 3134    | [Toutanova *et al*., 2015](https://www.aclweb.org/anthology/W15-4007/) |
| WN18          | 40943    | 18        | 141442  | 5000  | 5000    | [Bordes *et al*., 2013](https://arxiv.org/abs/1301.3485)     |
| WN11          | 38588    | 11        | 112581  | 2609  | 10544   | [Socher *et al*., 2013](https://dl.acm.org/doi/10.5555/2999611.2999715) |
| DBpedia50     | 49900    | 654       | 23288   | 399   | 10969   | [Shi *et al*., 2017](https://arxiv.org/abs/1711.03438)       |
| DBpedia500    | 517475   | 654       | 3102677 | 10000 | 1155937 |                                                              |
| Countries     | 271      | 2         | 1111    | 24    | 24      | [Bouchard *et al*., 2015](https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026) |
| FB13          | 75043    | 13        | 316232  | 5908  | 23733   | [Socher *et al*., 2013](https://dl.acm.org/doi/10.5555/2999611.2999715) |
| Kinsip        | 104      | 25        | 8544    | 1086  | 1074    | [Kemp *et al*., 2006](https://www.aaai.org/Papers/AAAI/2006/AAAI06-061.pdf) |
| Nations       | 14       | 55        | 1592    | 199   | 201     | [`ZhenfengLei/KGDatasets`](https://github.com/ZhenfengLei/KGDatasets) |
| NELL-995      | 75492    | 200       | 149678  | 543   | 3992    | [Nathani *et al*., 2019](https://arxiv.org/abs/1906.01195)   |
| UMLS          | 75492    | 135       | 5216    | 652   | 661     | [`ZhenfengLei/KGDatasets`](https://github.com/ZhenfengLei/KGDatasets) |
### EA datasets

| Datasets name    | Entities | Relations | Triples | Citation                                                     |
| ---------------- | -------- | --------- | ------- | ------------------------------------------------------------ |
| OpenEA supported | 15000    | 248       | 38265   | [Sun *et al*., 2020](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf) |

### ET datasets

| Datasets name | Entities | Relations | Triples | Types | Citation                                                     |
| ------------- | -------- | --------- | ------- | ----- | ------------------------------------------------------------ |
| FB15K-ET      | 15000    | 248       | 38265   | 3851  | [Moon *et al*., 2017](https://persagen.com/files/misc/Moon2017Learning.pdf) |




## Utils 📂

### Sampler

**Negative sampler:**

μKG includes several negative sampling methods to randomly generate negative examples.

- Uniform negative sampling:  This method replaces an entity in a triple or an alignment pair with another randomly-sampled entity to generate a negative example. It gives each entity the same replacement probability.
- Self-adversarial negative sampling: This method samples negative triples according to the current embedding model.
- Truncated negative sampling: This method seeks to generate hard negative examples.

**Path sampler:**
The Path sampler is to support some embedding models that are built by modeling the paths of KGs, such as IPTransE and RSN4EA. It can generate relational path like ***(e_1, r_1, e_2, r_2, e_3)***, entity path like ***(e_1, e_2, e_3)***, and relation path like ***(r_1, r_2)***.

**Subgraph sampler:**
The subgraph sampler is to support GNN-based embedding models like GCN-Align and AliNet. It can generate both first-order (i.e., one-hop) and high-order (i.e., multi-hop) neighborhood subgraphs of entities.

### Evaluator

**(joint) link prediction & entity typing:** It uses the energy function to compute the plausibility of a candidate triple. The implemented metrics for assessing the performance of embedding tasks include Hits@K, mean rank (MR) and mean reciprocal rank (MRR). The hyperparameter json file stored in `args` subfolder allows you to set Hits@K.

**entity alignment**: It provides several metrics to measure entity embedding similarities, such as the cosine, inner, Euclidean distance, and cross-domain similarity local scaling. The evaluation process can be accelerated using multiprocessing. 
### Multi-GPU and multi-processing computation

We use [Ray](https://www.ray.io/) to provide a uniform and easy-to-use interface for multi-GPU and multi-processing computation. The following figure shows our Ray-based implementation for parallel computing and the code snippet to use it. Users can set the number of CPUs or GPUs used for model training.

![image-20220507172436866](https://github.com/nju-websoft/muKG/blob/main/figs/ray.png)

To use the following command line to train your model with multi-GPU and multi-processing. Firstly check the number of resources on your machine (GPU or CPU), and then specify the number of parallels. The system will automatically allocate resources for each worker working in parallel. 

```bash
# When you run on one or more GPUs, use os.environ['CUDA_VISIBLE_DEVICES'] to set GPU id list first 
python main_args.py -t lp -m transe -o train -d data/FB15K -r gpu:2 -w 2  
```
## Running Experiments 🔬

### Instruction

We have provided the hyper-parameters of some models for critical experiments in the paper. These scripts can be founded in the folder [experiments](https://github.com/nju-websoft/muKG/tree/main/src/py/experiments). You can simply select the specific model in the corresponding Python file to reproduce experiments.  And we recommend you to check GPU resources when doing experiments on efficiency. Then add the following code to set GPU IDs for all RAY workers.

```bash
os.environ['CUDA_VISIBLE_DEVICES'] = "GPU IDs set"
```


### Efficiency of multi-GPU training

We give the evaluation results of the efficiency of the proposed library μKG here. The experiments were conducted on a server with an Intel Xeon Gold 6240 2.6GHz CPU, 512GB of memory and four NVIDIA Tesla V100 GPUs. The following figure compares the training time of RotatE and ConvE on FB15K-237 when using different numbers of GPUs. 

<div align=center>
<img src="https://github.com/nju-websoft/muKG/blob/main/figs/time.png" width="500" alt="image-20220508150812794" style="zoom: 50%;" />
</div>

### Training time comparison of different libraries

We further compare the training time used by μKG with LibKGE and PyKEEN. The backbone of μKG in this experiment is also PyTorch. We use the same hyper-parameter settings (e.g., batch size and maximum training epochs) for each model in the three libraries. The following table gives the training time of ConvE and RotatE on FB15K-237 with a single GPU for calculation.

| Models |  μKG  | LibKGE  | PyKEEN  |
| :----: | :---: |:-------:|:-------:|
| RotatE | 639 s | 3,260 s | 1,085 s |
| ConvE  | 824 s | 1,801 s |  961 s  |



## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

## Citation

```
@inproceedings{muKG,
  author    = {Xindi Luo and
  	       Zequn Sun and
               Wei Hu},
  title     = {μKG: A Library for Multi-source Knowledge Graph Embeddings and Applications},
  booktitle = {ISWC 2022},
  year      = {2022}
}
```




