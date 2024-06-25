from glob import glob
import pandas as pd
import json
from tqdm import tqdm
import yaml

def create_idmap(sample, root, idmappath, datapath):
    pattern = f'{root}/{datapath}/*.json*'
    print (pattern)
    files = glob(pattern)
    files.sort()
    idmap={'filename':[], 'min_adlr_id':[], 'max_adlr_id':[], 'min_id':[], 'max_id':[]}
    df=pd.DataFrame(idmap)
    
    idmappath = f'{root}/{idmappath}'
    
    
    files = files[:sample]
    print (len(files))
    idmap_name = f"{idmappath}/id_mapping.csv"
    adlr_name = f"{idmappath}/adlr_ids"
    
    global_id = 0
    files.sort()
    adlr_id_list = []
    for fname in tqdm(files):
        with open(fname, 'r') as f:
            data = [json.loads(line) for line in f]
            idmap['filename'].append(fname)
            idmap['min_adlr_id'].append(data[0]['adlr_id'])
            idmap['max_adlr_id'].append(data[-1]['adlr_id'])
            idmap['min_id'].append(global_id)
            idmap['max_id'].append(global_id + len(data) - 1)
            # update global_id
            global_id += len(data)
                           

    df=pd.DataFrame(idmap)
    df.to_csv(idmap_name, header=True, index=False)

    #with open(f"{idmappath}/adlr_ids", "w") as outfile:
    #    outfile.write("\n".join(adlr_id_list))

if __name__ == "__main__":
    config_file = "./configs.yaml"
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        
    root = params['root']
    sample = params['sample']
    idmappath = params['embeddings']['idmappath']
    datapath = params['embeddings']['datapath']
    create_idmap(sample, root, idmappath, datapath)
