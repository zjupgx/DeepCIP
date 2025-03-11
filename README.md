# DeepCIP
DeepCIP is a **Deep** learning method for **C**ircRNA **I**RES **P**rediction.

## System Requirments
DeepCIP needs to run on a Linux operating system (e.g. Debian 11.3) with the following software installed.
### Software Requirments:
* Python3.8
* Perl (Recommended v5.32.1)
* Anaconda


## The ViennaRNA package installation
RNAplfold from ViennaRNA version 2.5.1 is required to predict RNA secondary structure in DeepCIP. You need to install the ViennaRNA package before you start to use DeepCIP.

First, download the ViennaRNA package from [ViennaRNA-2.5.1.tar.gz](https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_5_x/ViennaRNA-2.5.1.tar.gz) .

Then, install ViennaRNA package:
```
tar -zxvf ViennaRNA-2.5.1.tar.gz
cd ViennaRNA-2.5.1
./configure
make
make install
```
For more details, see https://github.com/ViennaRNA/ViennaRNA

## Seqkit installation
DeepCIP relies on Seqkit to process RNA sequences (eg. subsequence segmentation).

You can download Seqkit from https://bioinf.shenwei.me/seqkit/download/

Reference: W Shen, S Le, Y Li*, F Hu*. SeqKit: a cross-platform and ultrafast toolkit for FASTA/Q file manipulation. PLOS ONE. doi:10.1371/journal.pone.0163962.

## Installation of DeepCIP and its environment
First, download the repository and create the environment.
```
git clone https://github.com/zjupgx/DeepCIP.git
cd ./DeepCIP
conda env create -f environment.yml
```
Then, activate the "DeepCIP_pytorch" environment.
```
conda activate DeepCIP_pytorch
```

## Usage
### Run DeepCIP for circRNA IRES prediction
The file prepared for prediction should be put into folder ./data. 
Then, type the following command to start the prediction, with parameters set according to your needs.

For example:
```
python DeepCIP.py -n example -i circires_case.fa
```
For more options:
```
python DeepCIP.py --help
```
```
usage: DeepCIP.py [-h] -n  -i  [-b] [-c] [-m] [-w] [-s] [-r ]

optional arguments:
  -h, --help           show this help message and exit
  -n , --data_name     The name of your input dataset.
  -i , --input_file    Input file for prediction. (*.fasta or *.fa file)
  -b , --batch_size    Batch size. (default=16) "--bs 16" means every 16 sampels constitute a prediction batch. This parameter affects the speed and results of the prediction. The larger the batch size, the faster the prediction, as far as your machine allows. (If the lengths of your input sequences vary greatly, it is recommended that you do not use a large batch size, or you can put sequences of similar lengths together for prediction)
  -c , --cut_off       Prediction threshold. (default=0.5)
  -m , --mode          The mode of prediction. (default=0) mode 0: Prediction directly on the input sequence. mode 1: The input sequence is partitioned by length w and interval s, and then the partitioned sequence is predicted. (w and s can be set by --w and --s, respectively)
                       mode 2:
  -w , --window_size   window size (default=174). See --mode description. It can be ignore when mode not is 1.
  -s , --step          step (default=50). See --mode description. It can be ignore when mode not is 1.
  -r  , --region       region of circRNA detection. e.g -r 1 12 for first 12 bases, -r -12 -1 for last 12 bases, -r 13 -1 for cutting first 12 bases. See --mode description. It can be ignore when mode not is 2.
  ```

The result will be saved as a csv format file.

## Datasets
Raw data used in our study can be avaliable in [GSE178718_Oligo_eGFP_expression](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE178718&format=file&file=GSE178718%5FOligo%5FeGFP%5Fexpression%2Exlsx) and [55k_oligos](https://bitbucket.org/alexeyg-com/irespredictor/src/v2/data/)
