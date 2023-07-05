import os
import shutil
import warnings
import datetime
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch_geometric.loader import DataLoader
from pytorch_lightning.trainer import seed_everything
from compiles.fusion_gnn import PL_Fusion
from compiles.create_graph_predict import RNAGraphDataset

def read_fa(path):
    res = {}
    records = list(SeqIO.parse(path, format='fasta'))
    for x in records:
        id = str(x.id)
        seq = str(x.seq)
        res[id] = seq
    return res


def DeepCIP_predict():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--data_name', default=None, help='The name of your input dataset.', metavar='', type=str, required=True)
    parser.add_argument('-i','--input_file', default=None, help='Input file for prediction. (*.fasta or *.fa file)', metavar='', type=str, required=True)
    parser.add_argument('-b','--batch_size', default=16, help='Batch size. (default=16)  "--bs 16" means every 32 sampels constitute a prediction batch. This parameter affects the speed and  results of the prediction. The larger the batch size, the faster the prediction, as far as your machine allows. (If the lengths of your input sequences vary greatly, it is recommended that you do not use a large batch size, or you can put sequences of similar lengths together for prediction)', metavar='', type=int)
    parser.add_argument('-c','--cut_off', default=0.5, help='Prediction threshold. (default=0.5)', metavar='', type=float)
    parser.add_argument('-m','--mode', default=0, choices=[0, 1, 2], help='The mode of prediction. (default=0)  mode 0: Prediction directly on the input sequence.  mode 1: The input sequence is partitioned by length w and interval s, and then the partitioned sequence is predicted. (w and s can be set by -w and -s, respectively)  mode 2: Predicting a particular region of interest in a single circRNA (-r to input the region in circRNA)', metavar='', type=int)
    parser.add_argument('-w','--window_size', default=174, help='window size (default=174). See --mode description. It can be ignore when mode not is 1.', metavar='', type=int)
    parser.add_argument('-s','--step', default=50, help='step (default=50). See --mode description. It can be ignore when mode not is 1.', metavar='', type=int)
    parser.add_argument('-r','--region', default=None, nargs=2, help='region of circRNA detection. e.g -r 1 12 for first 12 bases, -r -12 -1 for last 12 bases, -r 13 -1 for cutting first 12 bases. See --mode description. It can be ignore when mode not is 2.', metavar='', type=int)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    start = datetime.datetime.now()
    print(f'{start}: IRES prediction starting... \n')
    print('------------------------------------------------------------------')


    pred_struc_dir = './pred_struc/'
    if not os.path.exists(pred_struc_dir):
        os.mkdir(pred_struc_dir)
    else:
        print(pred_struc_dir + ' exist')

    print('Structure prediction...')

    if args.mode == 0:
        structure_pred0 = subprocess.call(['RNAplfold -W 150 -c 0.001 --noLP --auto-id --id-digits 10 <' + f'./data/{args.input_file}'], shell=True, stdout=subprocess.PIPE)
        input_file = args.input_file

    elif args.mode == 1:
        circrna_split = subprocess.call([f'seqkit sliding -s {args.step} -W {args.window_size} -C ./data/{args.input_file} > ./data/process_{args.input_file}'], \
            shell=True, stdout=subprocess.PIPE)
        seq2oneline = subprocess.call([f'seqkit seq ./data/process_{args.input_file} -w 0 > ./data/split_{args.input_file}'], shell=True, stdout=subprocess.PIPE)
        os.remove(f'./data/process_{args.input_file}')

        print('The sequence split is done!')

        structure_pred1 = subprocess.call(['RNAplfold -W 150 -c 0.001 --noLP --auto-id --id-digits 10 <' + f'./data/split_{args.input_file}'], shell=True, stdout=subprocess.PIPE)
        input_file = f'split_{args.input_file}'

    elif args.mode == 2:
        seq2oneline = subprocess.call([f'seqkit subseq -r {args.region[0]}:{args.region[1]} ./data/{args.input_file} > ./data/subseq_{args.input_file}'], shell=True, stdout=subprocess.PIPE)
        structure_pred2 = subprocess.call(['RNAplfold -W 150 -c 0.001 --noLP --auto-id --id-digits 10 <' + f'./data/subseq_{args.input_file}'], shell=True, stdout=subprocess.PIPE)
        input_file = f'subseq_{args.input_file}'

    for struc_file in os.listdir('./'):
        if struc_file.endswith('_dp.ps'):
            shutil.move(struc_file, pred_struc_dir + struc_file)

    print('Structure prediction done!')

    #get_fasta_seq_name
    #get_seq
    name_ls = []
    seq_ls = []
    res = read_fa('./data/' + f'{input_file}')
    for name in res.keys():
        name_ls.append(name)
        seq_ls.append(res[name].replace('U', 'T'))

    

    config = {
        'num_features': 4,
        'gcn_drop': 0.0,
        'graph_pool': 'sum',
        'output_dim': 1,
        'learning_rate': 1e-4,
        'batch_size': 256,
        'optim': 'Adam',
        'input_size': 13,
        'hidden_size': 64,
        'window_size': 3,
        'steps': 7,
        'sentence_nodes': 1,
        'slstm_drop': 0.0,
        'batch_first': False,
        'acti_func': 'ReLU',
        'conv_size1': 32,
        'conv_size2': 16,
        'kernel_size1': 4,
        'kernel_size2': 4
        
    }


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    data_test = RNAGraphDataset(dataset=args.data_name, rna_graph=pred_struc_dir,\
        xr=seq_ls, xs=np.array(seq_ls))

    loader = DataLoader(dataset=data_test, batch_size=args.batch_size)

    ckpt_paths1 = './model/Fusion/set1/version_0/checkpoints/epoch=49-step=1400.ckpt'
    ckpt_paths2 = './model/Fusion/set2/version_1/checkpoints/epoch=112-step=3164.ckpt'
    ckpt_paths3 = './model/Fusion/set3/version_3/checkpoints/epoch=37-step=1064.ckpt'


    preds_all = []

    preds1 = np.empty(0)
    seq_ckpt_paths1 = './model/S_LSTM/set1/version_0/checkpoints/epoch=40-step=1148.ckpt'
    struc_ckpt_paths1 = './model/GCN/set1/version_1/checkpoints/epoch=176-step=4956.ckpt'
    model1 = PL_Fusion.load_from_checkpoint(ckpt_paths1, config=config, seq_ckpt_path=seq_ckpt_paths1, struc_ckpt_path=struc_ckpt_paths1, map_location=device)
    model1.eval()

    preds2 = np.empty(0)
    seq_ckpt_paths2 = './model/S_LSTM/set2/version_0/checkpoints/epoch=18-step=532.ckpt'
    struc_ckpt_paths2 = './model/GCN/set2/version_1/checkpoints/epoch=5-step=168.ckpt'
    model2 = PL_Fusion.load_from_checkpoint(ckpt_paths2, config=config, seq_ckpt_path=seq_ckpt_paths2, struc_ckpt_path=struc_ckpt_paths2, map_location=device)
    model2.eval()

    preds3 = np.empty(0)
    seq_ckpt_paths3 = './model/S_LSTM/set3/version_0/checkpoints/epoch=19-step=560.ckpt'
    struc_ckpt_paths3 = './model/GCN/set3/version_1/checkpoints/epoch=1-step=56.ckpt'
    model3 = PL_Fusion.load_from_checkpoint(ckpt_paths3, config=config, seq_ckpt_path=seq_ckpt_paths3, struc_ckpt_path=struc_ckpt_paths3, map_location=device)
    model3.eval()

    with torch.no_grad():
        for test_data in loader:
            test_data = test_data.to(device)
            pred1 = model1(test_data).squeeze(-1)
            preds1 = np.append(preds1, pred1.cpu().numpy())
            pred2 = model2(test_data).squeeze(-1)
            preds2 = np.append(preds2, pred2.cpu().numpy())
            pred3 = model3(test_data).squeeze(-1)
            preds3 = np.append(preds3, pred3.cpu().numpy())


    preds_all.append(preds1)
    preds_all.append(preds2)
    preds_all.append(preds3)
    ensemble_probs = np.mean(preds_all, axis=0)

    ires_labels = []
    for i in ensemble_probs:
        if i < args.cut_off:
            ires_label = 'Non-CircIRES'
        else:
            ires_label = 'CircIRES'
        ires_labels.append(ires_label)

    out_dir = './results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_name = args.input_file.split('/')[-1].split('.')[0]
    outfile = f'{out_dir}/{out_name}_mode_{args.mode}.csv'

    result_df = pd.DataFrame()
    result_df['Sequence_name'] = name_ls
    result_df['Predict_probs'] = ensemble_probs
    result_df['IRES_label'] = ires_labels

    result_df.to_csv(outfile)


    shutil.rmtree(pred_struc_dir)
    for struc_file in os.listdir('./'):
        if struc_file.endswith('_dp.ps'):
            os.remove(struc_file)

    print('------------------------------------------------------------------')
    end = datetime.datetime.now()
    print(f'{end}: IRES prediction complete! \n')
    print('The prediction results were saved in {}'.format(outfile))

if __name__ == "__main__":
    DeepCIP_predict()
