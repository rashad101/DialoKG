# DialoKG: Knowledge-Structure Aware Task-Oriented Dialogue Generation
PyTorch code for NAACL 2022 paper: DialoKG: Knowledge-Structure Aware Task-Oriented Dialogue Generation [[PDF]](https://aclanthology.org/2022.findings-naacl.195.pdf).

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


### Installation (anaconda)
```commandline
conda create -n dialokg -y python=3.7 && source activate dialokg
pip install -r requirements.txt
chmod +x setup.sh
./setup.sh
```


### Citation
```
@inproceedings{rony-etal-2022-dialokg,
    title = "{D}ialo{KG}: Knowledge-Structure Aware Task-Oriented Dialogue Generation",
    author = "Rony, Md Rashad Al Hasan  and
      Usbeck, Ricardo  and
      Lehmann, Jens",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.195",
    pages = "2557--2571",
    abstract = "Task-oriented dialogue generation is challenging since the underlying knowledge is often dynamic and effectively incorporating knowledge into the learning process is hard. It is particularly challenging to generate both human-like and informative responses in this setting. Recent research primarily focused on various knowledge distillation methods where the underlying relationship between the facts in a knowledge base is not effectively captured. In this paper, we go one step further and demonstrate how the structural information of a knowledge graph can improve the system{'}s inference capabilities. Specifically, we propose DialoKG, a novel task-oriented dialogue system that effectively incorporates knowledge into a language model. Our proposed system views relational knowledge as a knowledge graph and introduces (1) a structure-aware knowledge embedding technique, and (2) a knowledge graph-weighted attention masking strategy to facilitate the system selecting relevant information during the dialogue generation. An empirical evaluation demonstrates the effectiveness of DialoKG over state-of-the-art methods on several standard benchmark datasets.",
}
```
### License
[MIT]()

### Contact
For further information, contact the corresponding author Md Rashad Al Hasan Rony ([email](mailto:rashad.research@gmail.com)).