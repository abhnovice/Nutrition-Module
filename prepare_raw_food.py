"""Download the user-specified HF dataset and prepare reproducible local image records."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'.hf_cache'
os.environ.setdefault('HF_HOME',str(CACHE))
os.environ.setdefault('HF_HUB_CACHE',str(CACHE/'hub'))
os.environ.setdefault('HF_DATASETS_CACHE',str(CACHE/'datasets'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN','1')
import json
import hashlib
import numpy as np
from PIL import ImageOps
from scipy.fft import dctn

REPO='ibrahimdaud/raw-food-recognition'
REVISION='e03922c47df5c8015616e3c88536883a1aacd10a'
DEST=ROOT/'Datasets'/'raw-food-recognition'


def canonical_label(label):
    return ' '.join(str(label).lower().replace('_',' ').replace('yoghurt','yogurt').split())


def image_hashes(im):
    rgb=im.convert('RGB')
    exact=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    small=np.asarray(rgb.convert('L').resize((32,32)),dtype=np.float32)
    values=dctn(small,norm='ortho')[:8,:8].ravel()
    bits=values>np.median(values[1:]);bits[0]=False
    perceptual=sum(int(b)<<i for i,b in enumerate(bits))
    return exact,f'{perceptual:016x}'


def main():
    from datasets import load_dataset
    DEST.mkdir(parents=True,exist_ok=True)
    revision_file=DEST/'source.json'
    revision=REVISION
    if revision_file.exists():
        source=json.loads(revision_file.read_text())
        if source.get('repository')!=REPO or source.get('revision')!=revision:
            raise ValueError('Local raw-food cache belongs to a different source revision.')
    else:
        revision_file.write_text(json.dumps({'repository':REPO,'revision':revision},indent=2))
    print(f'Loading {REPO} at {revision}',flush=True)
    ds=load_dataset(REPO,revision=revision,cache_dir=str(CACHE/'datasets'))
    print('Downloaded splits:',{k:len(v) for k,v in ds.items()},flush=True)
    records=[];bad=[]
    for split,subset in ds.items():
        folder=DEST/split;folder.mkdir(exist_ok=True)
        for index in range(len(subset)):
            try:
                row=subset[index]
                im=ImageOps.exif_transpose(row['image']).convert('RGB')
                exact,phash=image_hashes(im)
                target=folder/f'{index:06d}_{exact[:16]}.jpg'
                if not target.exists():
                    im.thumbnail((512,512));im.save(target,quality=95)
                records.append({'path':str(target.relative_to(ROOT)), 'label':canonical_label(row['label']),
                                'original_label':str(row['label']), 'source_split':split,'row':index,
                                'sha256':exact,'phash':phash})
            except (OSError,ValueError) as e:
                bad.append({'split':split,'row':index,'error':str(e)})
            if (index+1)%500==0:print(f'{split}: prepared {index+1}/{len(subset)}',flush=True)
    temporary=DEST/'manifest.pending.json'
    temporary.write_text(json.dumps({'repository':REPO,'revision':revision,'original_splits':{k:len(v) for k,v in ds.items()},'records':records,'unreadable_images':bad},indent=2))
    temporary.replace(DEST/'manifest.json')
    print('Prepared',len(records),'images; unreadable:',len(bad),flush=True)

if __name__=='__main__':main()
