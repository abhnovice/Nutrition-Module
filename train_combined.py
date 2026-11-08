"""Train and evaluate the shared dish/raw-food recognizer from audited local records."""
from pathlib import Path
import collections
import copy
from joblib import Parallel, delayed
import hashlib
import json
import importlib.metadata
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import fitness_food as app
from prepare_raw_food import image_hashes, canonical_label

RAW_MANIFEST=app.DATA/'raw-food-recognition'/'manifest.json'
PLAN=app.DATA/'combined_split_manifest.json'
CACHE=app.DATA/'combined_features.npz'


def plan_records(raw):
    """Keep source-train separate; divide raw validation by content group."""
    records=[]
    for split in ('train','valid','test'):
        for filename,label,labels in app.image_records(split):
            path=app.IMAGES/split/filename
            with Image.open(path) as im:exact,phash=image_hashes(im)
            records.append({'path':str(path.relative_to(app.ROOT)),'label':label,'labels':sorted(labels),
                            'split':split,'domain':'dish','sha256':exact,'phash':phash})
    groups=collections.defaultdict(list)
    for record in raw['records']:groups[record['sha256']].append(record)
    validation_by_label=collections.defaultdict(list)
    for digest,group in groups.items():
        if len({r['label'] for r in group})!=1:continue
        if any(r['source_split']!='train' for r in group):
            validation_by_label[group[0]['label']].append(digest)
    assignment={}
    for label,digests in validation_by_label.items():
        # Stable hash ordering independent of source row ordering.
        for index,digest in enumerate(sorted(digests,key=lambda d:hashlib.sha256(('42'+d).encode()).hexdigest())):
            assignment[digest]='valid' if index%2==0 else 'test'
    skipped=collections.Counter();source_duplicates=0
    for digest,group in groups.items():
        if len({r['label'] for r in group})!=1:
            skipped['conflicting_raw_labels']+=len(group);continue
        source_duplicates+=len(group)-1
        original=next((r for r in group if r['source_split']!='train'),group[0])
        records.append({'path':original['path'],'label':original['label'],'labels':[original['label']],
                        'split':assignment.get(digest,'train'),'domain':'raw','sha256':digest,'phash':original['phash']})
    # Evaluation takes precedence for duplicate content. Conflicting exact labels are dropped.
    exact_groups=collections.defaultdict(list)
    for r in records:exact_groups[r['sha256']].append(r)
    dedup=[];priority={'test':0,'valid':1,'train':2}
    for group in exact_groups.values():
        if len({r['label'] for r in group})>1:
            skipped['conflicting_cross_domain_labels']+=len(group);continue
        group.sort(key=lambda r:(priority[r['split']],r['path']))
        dedup.append(group[0]);skipped['identical_content_duplicates']+=len(group)-1
    # Remove near-identical same-class images across splits (evaluation has priority).
    accepted=[];phashes=collections.defaultdict(list)
    for r in sorted(dedup,key=lambda r:(priority[r['split']],r['path'])):
        code=int(r['phash'],16)
        if any(other_split!=r['split'] and (code^other).bit_count()<=4 for other,other_split in phashes[r['label']]):
            skipped['near_duplicate_cross_split']+=1;continue
        accepted.append(r);phashes[r['label']].append((code,r['split']))
    counts=collections.Counter(r['label'] for r in accepted if r['split']=='train')
    rare={label for label,n in counts.items() if n<3}
    supported=set(counts)-rare
    filtered=[r for r in accepted if r['label'] in supported]
    skipped['unsupported_or_rare_class_images']=len(accepted)-len(filtered)
    summary={f'{domain}_{split}':sum(r['domain']==domain and r['split']==split for r in filtered)
             for domain in ('dish','raw') for split in ('train','valid','test')}
    return {'source_repository':raw['repository'],'source_revision':raw['revision'],
            'source_splits':raw['original_splits'],'source_exact_duplicates':source_duplicates,
            'exclusions':dict(skipped),'rare_training_classes':sorted(rare),'counts':summary,
            'class_counts':{label:{split:sum(r['label']==label and r['split']==split for r in filtered) for split in ('train','valid','test')} for label in sorted(supported)},
            'split_policy':'Keep provided raw train as train; exact-content groups from provided validation are deterministically divided per class into valid/test. Drop exact label conflicts and cross-split same-label pHash distance <=4. Test takes precedence.',
            'limitations':'Perceptual hashing is a heuristic. It may exclude similar distinct photos and miss near duplicates. Raw test is carved from source validation, not an independent external benchmark.',
            'records':filtered}


def encode_plan(plan):
    file_state=[(r['path'],(app.ROOT/r['path']).stat().st_size,(app.ROOT/r['path']).stat().st_mtime_ns) for r in plan['records']]
    fingerprint=hashlib.sha256(json.dumps({'records':plan['records'],'file_state':file_state,'encoder':app.ENCODER_SHA256,'version':1},sort_keys=True).encode()).hexdigest()
    if CACHE.exists():
        with np.load(CACHE,allow_pickle=False) as d:
            if str(d['fingerprint'])==fingerprint:
                print('Using cached combined image embeddings.',flush=True)
                return d['x'].copy(),d['center'].copy()
    xs=[];centers=[]
    for index,r in enumerate(plan['records']):
        with Image.open(app.ROOT/r['path']) as im:
            from PIL import ImageOps
            im=ImageOps.exif_transpose(im).convert('RGB');w,h=im.size
            xs.append(app.deep_features(im))
            if r['split']!='train':centers.append(app.deep_features(im.crop((w*.15,h*.15,w*.85,h*.85))))
            else:centers.append(np.zeros(1280,dtype=np.float32))
        if (index+1)%500==0:print(f'Encoded {index+1}/{len(plan["records"])} combined photos',flush=True)
    x=np.asarray(xs,dtype=np.float32);center=np.asarray(centers,dtype=np.float32)
    temporary=CACHE.with_name('combined_features.pending.npz')
    np.savez_compressed(temporary,x=x,center=center,fingerprint=np.array(fingerprint));temporary.replace(CACHE)
    return x,center


def as_records(rows):return [(r['path'],r['label'],set(r['labels'])) for r in rows]


def domain_report(model,x,rows,accepted=None,votes=None):
    if votes is None:votes=app.model_votes(model,x)
    result={}
    for domain in ('dish','raw'):
        ids=np.array([i for i,r in enumerate(rows) if r['domain']==domain],dtype=int)
        if not len(ids):continue
        records=as_records([rows[i] for i in ids]);order=np.argsort(-votes[ids],axis=1)
        correct=np.array([model.classes_[index[0]] in r[2] for index,r in zip(order,records)])
        top3=[any(model.classes_[i] in r[2] for i in index[:3]) for index,r in zip(order,records)]
        grouped=collections.defaultdict(list)
        for value,record in zip(correct,records):grouped[record[1]].append(value)
        result[domain]={'raw':{'images':len(ids),'top1_accuracy':float(correct.mean()),'top3_accuracy':float(np.mean(top3)),
                            'macro_top1_accuracy':float(np.mean([np.mean(values) for values in grouped.values()]))}}
        if accepted is not None:result[domain]['selective']=app.selective_metrics(accepted[ids],votes[ids],model.classes_,records)
    return result


def per_class_report(model,x,rows,accepted):
    predicted=model.classes_[app.model_votes(model,x).argmax(axis=1)]
    result={}
    for label in sorted({r['label'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['label']==label]
        result[label]={'images':len(ids),'top1_accuracy':float(np.mean([predicted[i] in rows[i]['labels'] for i in ids])),
                       'predicted_labels':dict(collections.Counter(str(predicted[i]) for i in ids))}
        kept=[i for i in ids if accepted[i]]
        result[label]['accepted_images']=len(kept)
        result[label]['accepted_top1_accuracy']=float(np.mean([predicted[i] in rows[i]['labels'] for i in kept])) if kept else None
    return result



def combined_threshold(confidence,stable,distances,limit,correct,domains):
    for threshold in np.unique(np.r_[.5,confidence[confidence>=.5]]):
        kept=(confidence>=threshold)&stable&(distances<=limit)
        eligible=True
        for domain in ('dish','raw'):
            mask=kept&(domains==domain)
            if int(mask.sum())<15 or float(correct[mask].mean())<.7:
                eligible=False;break
        if eligible:return float(threshold)
    return 1.01



def calibration_fold(parameters,raw_classes,x,y,fit_ids,predict_ids):
    model=app.make_image_model(parameters['kind'],parameters['C'],raw_classes=raw_classes,
                               routing_threshold=parameters.get('routing_threshold',.5))
    model.fit(x[fit_ids],y[fit_ids])
    return predict_ids,model.classes_,model.decision_function(x[predict_ids])


def main():
    raw=json.loads(RAW_MANIFEST.read_text())
    print('Auditing image splits and label conflicts…',flush=True)
    plan=plan_records(raw);PLAN.write_text(json.dumps(plan,indent=2))
    print(json.dumps({'counts':plan['counts'],'exclusions':plan['exclusions']},indent=2),flush=True)
    features,centers=encode_plan(plan)
    with np.load(CACHE,allow_pickle=False) as cached_features:features_fingerprint=str(cached_features['fingerprint'])
    library_versions={name:importlib.metadata.version(name) for name in ('numpy','scikit-learn','onnxruntime','onnx','datasets','Pillow')}
    joint_signature=hashlib.sha256(json.dumps({'parameters':app.make_image_model('deep_svc',3).get_params(),'libraries':library_versions},sort_keys=True).encode()).hexdigest()
    rows={s:[r for r in plan['records'] if r['split']==s] for s in ('train','valid','test')}
    indices={s:np.array([i for i,r in enumerate(plan['records']) if r['split']==s]) for s in rows}
    x=features[indices['train']];y=np.array([r['label'] for r in rows['train']])
    vx=features[indices['valid']];vc=centers[indices['valid']];vr=as_records(rows['valid'])
    best=None;trials=[]
    previous_report={}
    report_path=app.ROOT/'training_report.json'
    if report_path.exists():previous_report=json.loads(report_path.read_text())
    cached=previous_report.get('validation_trials',[]) if (previous_report.get('features_fingerprint')==features_fingerprint and previous_report.get('joint_signature')==joint_signature and previous_report.get('data')=={k:v for k,v in plan.items() if k!='records'} and previous_report.get('encoder',{}).get('sha256')==app.ENCODER_SHA256) else []
    cached={t['C']:t for t in cached if t.get('kind','deep_svc')=='deep_svc'}
    for c in (1,3,10):
        if c in cached:
            trial={**cached[c],'kind':'deep_svc'};model=None
            print('Reusing deterministic joint validation trial for unchanged data:',c,flush=True)
        else:
            model=app.make_image_model('deep_svc',c);model.fit(x,y)
            domain=domain_report(model,vx,rows['valid'])
            score=float(np.mean([r['raw']['macro_top1_accuracy'] for r in domain.values()]))
            trial={'kind':'deep_svc','C':c,'balanced_domain_macro_validation_accuracy':score,'domains':domain}
        print(json.dumps(trial),flush=True);trials.append(trial)
        if best is None or trial['balanced_domain_macro_validation_accuracy']>best[0]['balanced_domain_macro_validation_accuracy']:best=trial,model
    raw_classes=tuple(sorted({r['label'] for r in rows['train'] if r['domain']=='raw'}))
    print('Comparing separate raw/dish heads with a learned router…',flush=True)
    hierarchy=app.make_image_model('hierarchical_svc',3,raw_classes=raw_classes)
    hierarchy.fit(x,y)
    cleaned_heads={}
    for name,head in [('dish',hierarchy.dish_model_),('raw',hierarchy.raw_model_)]:
        ids=np.array([i for i,r in enumerate(rows['valid']) if r['domain']==name])
        cleaned_heads[name]=app.metrics(head,vx[ids],as_records([rows['valid'][i] for i in ids]))
    print('Independent heads on cleaned validation:',json.dumps(cleaned_heads),flush=True)
    base_scores=hierarchy.decision_function(vx)
    raw_columns=np.isin(hierarchy.classes_,raw_classes)
    for routing in (.3,.5,.7,.85,.95):
        hierarchy.routing_threshold=routing
        offset=-np.log(routing/(1-routing))/2
        shifted=base_scores.copy();shifted[:,raw_columns]+=offset;shifted[:,~raw_columns]-=offset
        shifted-=shifted.max(axis=1,keepdims=True);probabilities=np.exp(shifted);probabilities/=probabilities.sum(axis=1,keepdims=True)
        domains=domain_report(hierarchy,vx,rows['valid'],votes=probabilities)
        score=float(np.mean([d['raw']['macro_top1_accuracy'] for d in domains.values()]))
        trial={'kind':'hierarchical_svc','C':3,'routing_threshold':routing,'balanced_domain_macro_validation_accuracy':score,'domains':domains}
        trials.append(trial);print(json.dumps(trial),flush=True)
        if score>best[0]['balanced_domain_macro_validation_accuracy']:best=trial,copy.copy(hierarchy)
    selected,model=best
    if model is None:
        model=app.make_image_model('deep_svc',selected['C']);model.fit(x,y)
    print('Fitting correctness estimator using training-only out-of-fold predictions…',flush=True)
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=42).split(x,y))
    fold_results=Parallel(n_jobs=2)(delayed(calibration_fold)(selected,raw_classes,x,y,fit_ids,predict_ids) for fit_ids,predict_ids in folds)
    scores=np.zeros((len(y),len(model.classes_)),dtype=np.float64)
    for ids,classes,values in fold_results:
        scores[np.ix_(ids,np.searchsorted(model.classes_,classes))]=values
    scores-=scores.max(axis=1,keepdims=True);oof=np.exp(scores);oof/=oof.sum(axis=1,keepdims=True)
    correct=(model.classes_[oof.argmax(axis=1)]==y).astype(int)
    calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
    calibrator.fit(app.risk_features(oof),correct)
    scale=calibrator.steps[0][1];logistic=calibrator.steps[1][1]
    coefficients=logistic.coef_[0]/scale.scale_;intercept=float(logistic.intercept_[0]-coefficients@scale.mean_)
    scaler,neighbors,limit=app.novelty_model(x)
    _,vv,distances,stable,confidence=app.rejection_checks(model,vx,vc,scaler,neighbors,limit,0,coefficients,intercept)
    vcorrect=np.array([model.classes_[i] in r[2] for i,r in zip(vv.argmax(axis=1),vr)])
    # A shared threshold must satisfy the acceptance target in BOTH domains.
    thresholds=[]
    for domain in ('raw','dish'):
        mask=np.array([r['domain']==domain for r in rows['valid']])
        thresholds.append(app.choose_rejection_threshold(confidence[mask],stable[mask],distances[mask],limit,vcorrect[mask]))
    threshold=combined_threshold(confidence,stable,distances,limit,vcorrect,np.array([r['domain'] for r in rows['valid']]))
    va,vv,_,_,_=app.rejection_checks(model,vx,vc,scaler,neighbors,limit,threshold,coefficients,intercept)
    print('Selection frozen. Evaluating dish and raw-food test photos…',flush=True)
    tx=features[indices['test']];tc=centers[indices['test']];tr=as_records(rows['test'])
    ta,tv,_,_,_=app.rejection_checks(model,tx,tc,scaler,neighbors,limit,threshold,coefficients,intercept)
    baseline_summary=previous_report.get('baseline')
    with np.load(app.MODEL,allow_pickle=False) as previous:
        if 'raw_classes' not in previous:
            baseline=app.make_image_model(str(previous['kind']),float(previous['c']))
            baseline.fit(previous['x'],previous['y'])
            dish_ids=np.array([i for i,r in enumerate(rows['test']) if r['domain']=='dish'])
            milk_ids=np.array([i for i,r in enumerate(rows['test']) if r['label']=='milk'])
            baseline_summary={'dish_test':app.metrics(baseline,tx[dish_ids],as_records([rows['test'][i] for i in dish_ids]))}
            if len(milk_ids):
                milk_predictions=baseline.classes_[app.model_votes(baseline,tx[milk_ids]).argmax(axis=1)]
                baseline_summary['untrained_milk_predictions']=dict(collections.Counter(str(p) for p in milk_predictions))
    report={'schema_version':4,'library_versions':library_versions,'features_fingerprint':features_fingerprint,'joint_signature':joint_signature,'training_images':len(x),'classes':len(model.classes_),
        'baseline':baseline_summary,
        'raw_classes':sorted({r['label'] for r in rows['train'] if r['domain']=='raw'}),
        'dish_classes':sorted({r['label'] for r in rows['train'] if r['domain']=='dish'}),
        'data':{k:v for k,v in plan.items() if k!='records'},'cleaned_independent_heads_validation':cleaned_heads,'validation_trials':trials,'selected':selected,
        'validation':domain_report(model,vx,rows['valid'],va,vv),'test':domain_report(model,tx,rows['test'],ta,tv),
        'test_per_class':per_class_report(model,tx,rows['test'],ta),
        'rejection':{'correctness_threshold':threshold,'domain_thresholds':thresholds,'distance_limit':limit,
                     'training_oof_top1_accuracy':float(correct.mean()),'validation_brier':float(np.mean((confidence-vcorrect)**2)),
                     'policy':'Validation-only shared threshold (minimum 50% estimate, >=15 accepted examples and >=70% accepted accuracy in each domain); require center stability and training novelty distance.'},
        'encoder':{'name':'Frozen MobileNetV2','sha256':app.ENCODER_SHA256,'dimensions':1280},
        'selection':'Mean of raw-food and dish macro class validation accuracy; training-only 3-fold correctness calibration. Test not used for fitting, model selection or threshold selection in this run.',
        'limitations':'Single dominant-item classification; no ingredient segmentation. Both dish and raw test splits have been inspected across development rounds and are not fresh external blind benchmarks. Source annotations are classification labels, not boxes or masks. Dataset does not provide ingredient nutrition.'}
    # Preserve the previous working model before atomic activation of this version.
    if app.MODEL.exists() and not (app.ROOT/'model_before_raw_food.npz').exists():
        shutil.copyfile(app.MODEL,app.ROOT/'model_before_raw_food.npz')
        if (app.ROOT/'training_report.json').exists():
            shutil.copyfile(app.ROOT/'training_report.json',app.ROOT/'training_report_before_raw_food.json')
    temporary=app.ROOT/'model.pending.npz'
    np.savez_compressed(temporary,x=x,y=y,kind=np.array(selected['kind']),c=np.array(selected['C']),routing_threshold=np.array(selected.get('routing_threshold',.5)),version=np.array(2),
        feature_kind=np.array('mobilenetv2'),encoder_sha256=np.array(app.ENCODER_SHA256),
        probability_threshold=np.array(threshold),distance_limit=np.array(limit),safety_version=np.array(2),
        risk_coefficients=coefficients,risk_intercept=np.array(intercept),
        raw_classes=np.array(report['raw_classes']),source_revision=np.array(raw['revision']),split_manifest_sha256=np.array(hashlib.sha256(PLAN.read_bytes()).hexdigest()))
    temporary.replace(app.MODEL)
    (app.ROOT/'training_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('training_images','classes','validation','test','rejection')},indent=2),flush=True)
    for label in ('milk','doodhpak','kheer'):
        print(label,json.dumps(report['test_per_class'].get(label)),flush=True)

if __name__=='__main__':main()
