import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
from fairseq.models.transformer import TransformerModel


dev_data_path = sys.argv[1] # dev data for computing inter_dis and intra_dis
model_path = sys.argv[2] # baseline model path
data_path = sys.argv[3] # spm model, data-bin

## load data
dev_path=dev_data_path
file_names=os.listdir(dev_path)
all_data={}
for name in file_names:
    if 'valid' not in name or 'en' not in name :
        continue
    file=os.path.join(dev_path, name)
    with open(file, 'r', encoding='utf-8') as f:
        all_data[name[-8:]]=[ line.strip() for line in f.readlines()][:100]

all_tasks = "de-en,fr-en,nl-en,zh-en,ar-en,ru-en".split(',')
langs="de,en,fr,nl,zh,ar,ru".split(',')

def teacher_forcing_forward(model, src_sent, tgt_sent, src_lang, tgt_lang, features_only=False):
    """example:
    src_tokens[0]: tensor([40001,  3481, 21269, 39830,     2], device='cuda:0')
    prev_output[0]: tensor([    2, 40004,  3488, 39830], device='cuda:0')
    """
    # assert isinstance(model, GeneratorHubInterface)
    src_langid=torch.LongTensor([model.src_dict.index('__{}__'.format(src_lang))])
    tgt_langid=torch.LongTensor([model.src_dict.index('__{}__'.format(tgt_lang))])
    src_tensor = torch.cat((src_langid, model.encode(src_sent))).cuda()
    tgt_tensor = torch.cat((tgt_langid, model.encode(tgt_sent))).cuda()
    prev_output = tgt_tensor.clone()
    prev_output[0] = tgt_tensor[-1]
    prev_output[1:] = tgt_tensor[:-1]
    net_input = {
            "src_tokens": src_tensor.unsqueeze(0),
            "prev_output_tokens": prev_output.unsqueeze(0),
            "src_lengths": src_tensor.size(),
            "features_only": features_only,
        }
    output = model.models[0](**net_input)
    return output[0].detach()


contextual_features={}
inter_dis = {}
intra_dis = {}
if os.path.isfile("./inter_dis.npy"):
    intra_dis = np.load('./intra_dis.npy', allow_pickle=True).item()
    inter_dis = np.load('./inter_dis.npy', allow_pickle=True).item()
    print("step, CWRsID")
    for key in intra_dis.keys():
        print("{},{}".format(key, inter_dis[key].item()/intra_dis[key].item()))
else:
    with torch.no_grad():
        for step in range(21): # 21 checkpoint
            if step == 0:
                opus100_model = TransformerModel.from_pretrained(
                    f'{model_path}',
                    checkpoint_file='checkpoint_best.pt',
                    task="translation_multi_simple_epoch",
                    data_name_or_path=f'{data_path}/data-bin',
                    bpe='sentencepiece',
                    sentencepiece_model=f'{data_path}/spm_64k.model',
                    langs='af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu'.split(',')
                ).cuda()
            else:
                step = step * 500
                opus100_model = TransformerModel.from_pretrained(
                    f'{model_path}/continue_train.continue_step10000.negative_loss_lambda1.0.translation_loss_lambda1.0.lr0.00005.wu1',
                    checkpoint_file='checkpoint_1_{}.pt'.format(step),
                    task="translation_multi_simple_epoch",
                    data_name_or_path=f'{data_path}/data-bin',
                    bpe='sentencepiece',
                    sentencepiece_model=f'{data_path}/spm_64k.model',
                    langs='af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu'.split(',')
                ).cuda()
            contextual_features[step] = {}
            intra_dis[step] = {}
            inter_dis[step] = {}
            for task in all_tasks:
                contextual_features[step][task] = {}
                src, tgt = task.split('-')
                if src == 'en':
                    src_data = all_data[task + "." + src]
                    tgt_data = all_data[task + "." + tgt]
                else:
                    src_data = all_data[tgt + '-' + src + "." + src]
                    tgt_data = all_data[tgt + '-' + src + "." + tgt]
                
                for tgt_lang in langs:
                    if tgt_lang == src or tgt_lang == tgt:
                        continue
                    contextual_features[step][task][src + '-' + tgt_lang] = []

                    for id, sentence in enumerate(src_data):
                        contextual_feature = teacher_forcing_forward(
                            opus100_model, 
                            sentence, 
                            tgt_data[id], 
                            src, 
                            tgt_lang,
                            ).cpu()
                        contextual_features[step][task][src + '-' + tgt_lang].append(contextual_feature.squeeze(0))

                # inter language distance
                inter_distances = {}
                keys = list(contextual_features[step][task].keys())
                for key in contextual_features[step][task].keys():
                    contextual_features[step][task][key] = torch.cat(contextual_features[step][task][key], dim=0).cuda()

                for i in range(len(keys)):
                    for j in range(len(keys) - i -1):
                        j = j + i + 1
                        inter_distances[keys[i] + '&' + keys[j]] = F.pairwise_distance(contextual_features[step][task][keys[i]], contextual_features[step][task][keys[j]], p=2).mean().cpu()

                avg_dis = 0
                for key in inter_distances.keys():
                    avg_dis += inter_distances[key]

                avg_dis = avg_dis / len(list(inter_distances.keys()))
                inter_dis[step][task] = avg_dis

                # intra language distance
                itra_distances = {}
                keys = list(contextual_features[step][task].keys())
                for i in range(len(keys)):
                    cur_contextual_features = contextual_features[step][task][keys[i]]
                    mean_features = cur_contextual_features.mean(dim=0).expand(cur_contextual_features.size())
                    itra_distances[keys[i]] = F.pairwise_distance(cur_contextual_features, mean_features, p=2).mean().cpu()

                avg_dis = 0
                for key in itra_distances.keys():
                    avg_dis += itra_distances[key]
                avg_dis = avg_dis / len(list(itra_distances.keys()))
                intra_dis[step][task] = avg_dis
                del contextual_features[step][task]
            
            avg_dis = 0
            for task in intra_dis[step].keys():
                avg_dis += intra_dis[step][task]
            intra_dis[step] = avg_dis
            avg_dis = 0
            for task in inter_dis[step].keys():
                avg_dis += inter_dis[step][task]
            inter_dis[step] = avg_dis

            print("{},{}".format(step, inter_dis[step].item()/intra_dis[step].item()))
            np.save('./inter_dis.npy', inter_dis)
            np.save('./intra_dis.npy', intra_dis)
            del contextual_features[step]
