"""Local food ML, nutrition lookup, barcode decoding, and a tiny test HTTP server."""
import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import urllib.error
import re
import threading
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageOps
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
import hashlib
import external_services as services

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Datasets'
IMAGES = DATA / 'Indian Food Yolov5.v3i.tensorflow'
MODEL = ROOT / 'model.npz'
Image.MAX_IMAGE_PIXELS = 20_000_000


def normalize(s):
    return ' '.join(re.findall(r'[a-z0-9]+', s.lower()))


def features(im):
    im = ImageOps.exif_transpose(im).convert('RGB').resize((64, 64))
    a = np.asarray(im, dtype=np.float32) / 255
    hsv = np.asarray(im.convert('HSV'))
    hist = np.concatenate([np.histogram(hsv[:, :, c], bins=16, range=(0, 256), density=True)[0] for c in range(3)])
    # Spatial color and gradient summaries provide a small CPU-only ML baseline.
    small = np.asarray(im.resize((12, 12)), dtype=np.float32).ravel() / 255
    gray = a.mean(axis=2)
    gy, gx = np.gradient(gray)
    angles = np.arctan2(gy, gx)
    magnitude = np.hypot(gx, gy)
    hog = np.concatenate([np.histogram(angles[y:y+16, x:x+16], bins=8, range=(-np.pi, np.pi), weights=magnitude[y:y+16, x:x+16])[0] for y in range(0,64,16) for x in range(0,64,16)])
    return np.concatenate([hist * 256, small, hog]).astype(np.float32)


ENCODER = ROOT / 'mobilenetv2-12.onnx'
ENCODER_SHA256 = 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5'
_ENCODER_SESSION = None
_ENCODER_LOCK = threading.Lock()


def deep_features(im):
    global _ENCODER_SESSION
    if _ENCODER_SESSION is None:
        with _ENCODER_LOCK:
            if _ENCODER_SESSION is None:
                if not ENCODER.exists() or hashlib.sha256(ENCODER.read_bytes()).hexdigest() != ENCODER_SHA256:
                    raise ValueError('Missing or changed MobileNet encoder. See README for the verified download.')
                import onnx
                import onnxruntime as ort
                ort.disable_telemetry_events()
                model=onnx.load(ENCODER)
                pooled=next(n.output[0] for n in reversed(model.graph.node) if n.op_type=='GlobalAveragePool')
                del model.graph.output[:]
                model.graph.output.append(onnx.helper.make_tensor_value_info(pooled,onnx.TensorProto.FLOAT,[None,1280,1,1]))
                options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
                options.log_severity_level=3
                _ENCODER_SESSION=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
    im=ImageOps.exif_transpose(im).convert('RGB')
    # Preserve the complete frame; the separate center crop is used for stability checks.
    arr=np.asarray(im.resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255
    arr=(arr-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    tensor=np.ascontiguousarray(arr.transpose(2,0,1)[None])
    vector=_ENCODER_SESSION.run(None,{_ENCODER_SESSION.get_inputs()[0].name:tensor})[0].ravel()
    return (vector/max(float(np.linalg.norm(vector)),1e-12)).astype(np.float32)


def visual_features(im, kind='handcrafted'):
    return deep_features(im) if kind=='mobilenetv2' else features(im)


def samples(split):
    with (IMAGES / split / '_annotations.csv').open() as f:
        for row in csv.DictReader(f):
            yield row


def image_records(split):
    """Group boxes per image; use largest annotated area as primary food label."""
    grouped = {}
    for row in samples(split):
        grouped.setdefault(row['filename'], []).append(row)
    records = []
    for filename, rows in sorted(grouped.items()):
        areas = {}
        for r in rows:
            area = max(0, int(r['xmax'])-int(r['xmin'])) * max(0, int(r['ymax'])-int(r['ymin']))
            areas[r['class']] = areas.get(r['class'], 0) + area
        records.append((filename, max(areas, key=areas.get), set(areas)))
    return records


def split_features(split, feature_kind='handcrafted'):
    records = image_records(split)
    xs = []
    for index, (filename, _, _) in enumerate(records):
        with Image.open(IMAGES / split / filename) as im:
            xs.append(visual_features(im,feature_kind))
        if feature_kind == 'mobilenetv2' and (index+1)%500==0:
            print(f'{split}: encoded {index+1}/{len(records)} photos',flush=True)
    return np.asarray(xs), np.asarray([r[1] for r in records]), records



class HierarchicalFoodClassifier(ClassifierMixin, BaseEstimator):
    """Separate raw/dish heads so new ingredient labels need not crowd out dishes."""
    def __init__(self,c=3,raw_classes=(),routing_threshold=.5):
        self.c=c;self.raw_classes=raw_classes;self.routing_threshold=routing_threshold

    def fit(self,x,y):
        y=np.asarray(y);self.classes_=np.unique(y)
        raw=np.isin(y,self.raw_classes)
        if not raw.any() or raw.all():raise ValueError('Hierarchical training needs raw and dish examples.')
        self.router_=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42))
        self.router_.fit(x,raw.astype(int))
        self.raw_model_=make_image_model('deep_svc',self.c);self.raw_model_.fit(x[raw],y[raw])
        self.dish_model_=make_image_model('deep_svc',self.c);self.dish_model_.fit(x[~raw],y[~raw])
        self.n_features_in_=x.shape[1]
        return self

    def decision_function(self,x):
        probability=np.clip(np.asarray(self.router_.predict_proba(x)[:,1],dtype=np.float64),1e-8,1-1e-8)
        cutoff=self.routing_threshold
        gate=np.log(probability/(1-probability))-np.log(cutoff/(1-cutoff))
        result=np.empty((len(x),len(self.classes_)),dtype=np.float64)
        for head,offset in [(self.raw_model_,gate/2),(self.dish_model_,-gate/2)]:
            scores=head.decision_function(x)
            if scores.ndim==1:scores=np.column_stack([-scores,scores])
            scores=scores-scores.max(axis=1,keepdims=True)+offset[:,None]
            result[:,np.searchsorted(self.classes_,head.classes_)]=scores
        return result

    def predict(self,x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]


def make_image_model(kind, c=10, raw_classes=(), routing_threshold=.5):
    if kind=='hierarchical_svc':
        return HierarchicalFoodClassifier(c,raw_classes,routing_threshold)
    if kind == 'deep_svc':
        return SVC(C=c, kernel='rbf', class_weight='balanced', decision_function_shape='ovr',random_state=42)
    if kind == 'svc':
        return make_pipeline(StandardScaler(), SVC(C=c, kernel='rbf', class_weight='balanced', decision_function_shape='ovr', random_state=42))
    return ExtraTreesClassifier(n_estimators=160, min_samples_leaf=2, class_weight='balanced', n_jobs=2, random_state=42)


def model_votes(model, xs):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(xs)
    scores = model.decision_function(xs)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    votes = np.exp(scores)
    return votes / votes.sum(axis=1, keepdims=True)


def metrics(model, xs, records):
    votes = model_votes(model, xs)
    order = np.argsort(-votes, axis=1)
    top1 = [model.classes_[idx[0]] in r[2] for idx, r in zip(order, records)]
    top3 = [any(model.classes_[i] in r[2] for i in idx[:3]) for idx, r in zip(order, records)]
    return {'images':len(records), 'top1_accuracy':float(np.mean(top1)), 'top3_accuracy':float(np.mean(top3))}


def split_audit():
    sources, hashes = {}, {}
    for split in ('train', 'valid', 'test'):
        sources[split], hashes[split] = set(), set()
        for filename, _, _ in image_records(split):
            sources[split].add(filename.split('.rf.')[0])
            with Image.open(IMAGES / split / filename) as im:
                rgb = im.convert('RGB')
                hashes[split].add(hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest())
    return {a+'_'+b:{'shared_source_names':len(sources[a]&sources[b]), 'identical_decoded_images':len(hashes[a]&hashes[b])}
            for a,b in [('train','valid'),('train','test'),('valid','test')]}


def train():
    """Train the current local transfer-learning model and rejection policy."""
    refine()


def novelty_model(xs, distance_limit=None):
    scaler = StandardScaler().fit(xs)
    scaled = scaler.transform(xs)
    neighbors = NearestNeighbors(n_neighbors=2, n_jobs=2).fit(scaled)
    if distance_limit is None:
        distances = neighbors.kneighbors(scaled)[0][:,1] / np.sqrt(xs.shape[1])
        distance_limit=float(np.quantile(distances, .99))
    return scaler, neighbors, float(distance_limit)


def risk_features(votes):
    ordered=np.sort(votes,axis=1)
    entropy=-(votes*np.log(np.maximum(votes,1e-12))).sum(axis=1) / np.log(votes.shape[1])
    return np.column_stack([ordered[:,-1],ordered[:,-1]-ordered[:,-2],entropy])


def correctness_estimate(votes, coefficients, intercept):
    logits=risk_features(votes) @ coefficients + intercept
    return 1/(1+np.exp(-np.clip(logits,-40,40)))


def rejection_checks(model, xs, center_xs, scaler, neighbors, distance_limit, probability_threshold, coefficients, intercept):
    votes = model_votes(model, xs)
    center_votes = model_votes(model, center_xs)
    top = votes.argmax(axis=1)
    distances = neighbors.kneighbors(scaler.transform(xs), n_neighbors=1)[0][:,0] / np.sqrt(xs.shape[1])
    confidence=correctness_estimate(votes,coefficients,intercept)
    accepted = ((confidence >= probability_threshold) & (top == center_votes.argmax(axis=1)) & (distances <= distance_limit))
    return accepted, votes, distances, top == center_votes.argmax(axis=1), confidence


def selective_metrics(accepted, probs, classes, records):
    correct = np.array([classes[i] in r[2] for i,r in zip(probs.argmax(axis=1), records)])
    count = int(accepted.sum())
    return {'images':len(records), 'accepted_images':count, 'coverage':float(accepted.mean()),
            'accepted_top1_accuracy':float(correct[accepted].mean()) if count else None,
            'accepted_errors':int((~correct & accepted).sum()), 'rejected_images':int((~accepted).sum())}


def center_features(split, feature_kind='handcrafted'):
    result = []
    for filename, _, _ in image_records(split):
        with Image.open(IMAGES / split / filename) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            w,h=im.size
            result.append(visual_features(im.crop((w*.15,h*.15,w*.85,h*.85)),feature_kind))
    return np.asarray(result)



def choose_rejection_threshold(confidence, stable, distances, distance_limit, correct):
    # Always reject below 50% estimated top-1 reliability; tune only on validation.
    for threshold in np.unique(np.r_[.5, confidence[confidence>=.5]]):
        keep=(confidence>=threshold)&stable&(distances<=distance_limit)
        if int(keep.sum())>=15 and float(correct[keep].mean())>=.70:
            return float(threshold)
    return 1.01


def refine():
    if (DATA/'raw-food-recognition'/'manifest.json').exists():
        from train_combined import main as train_combined
        train_combined(); return
    print('Loading training features and validation photos…', flush=True)
    feature_kind='mobilenetv2'
    xs,ys,_=split_features('train',feature_kind)
    vx, _, records = split_features('valid',feature_kind)
    centers = center_features('valid',feature_kind)
    trials=[];best=None
    for c in (.3, 1, 3, 10):
        model=make_image_model('deep_svc',c)
        model.fit(xs,ys)
        result={'kind':'deep_svc','C':c,**metrics(model,vx,records)}
        trials.append(result); print(json.dumps(result),flush=True)
        if best is None or result['top1_accuracy'] > best[0]['top1_accuracy']:
            best=result,model
    selected,model=best
    print('Learning correctness estimates from training-only out-of-fold predictions…',flush=True)
    scores=cross_val_predict(make_image_model('deep_svc',selected['C']),xs,ys,
                cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42),method='decision_function',n_jobs=2)
    scores-=scores.max(axis=1,keepdims=True)
    oof_votes=np.exp(scores);oof_votes/=oof_votes.sum(axis=1,keepdims=True)
    oof_correct=(model.classes_[oof_votes.argmax(axis=1)]==ys).astype(int)
    calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
    calibrator.fit(risk_features(oof_votes),oof_correct)
    standard=calibrator.steps[0][1];logistic=calibrator.steps[1][1]
    coefficients=logistic.coef_[0]/standard.scale_
    intercept=float(logistic.intercept_[0]-coefficients@standard.mean_)
    scaler,neighbors,distance_limit=novelty_model(xs)
    _,votes,distances,stable,confidence=rejection_checks(model,vx,centers,scaler,neighbors,distance_limit,0,coefficients,intercept)
    top_correct=np.array([model.classes_[i] in r[2] for i,r in zip(votes.argmax(axis=1),records)])
    threshold=choose_rejection_threshold(confidence,stable,distances,distance_limit,top_correct)
    va,_,_,_,_=rejection_checks(model,vx,centers,scaler,neighbors,distance_limit,threshold,coefficients,intercept)
    print('Evaluating frozen model and rejection policy on test…',flush=True)
    tx,_,test_records=split_features('test',feature_kind);tc=center_features('test',feature_kind)
    ta,tp,_,_,test_confidence=rejection_checks(model,tx,tc,scaler,neighbors,distance_limit,threshold,coefficients,intercept)
    safety={'correctness_threshold':threshold,'distance_limit':distance_limit,
            'policy':'Correctness floor 50%; validation-selected threshold: >=70% empirical accepted accuracy, >=15 accepted photos; reject if no threshold qualifies. Also require center-crop agreement and training nearest-neighbor distance <=99th percentile.',
            'calibration':'Logistic top-1 correctness estimator fitted to training-only 3-fold out-of-fold predictions. Does not change dish ranking.',
            'training_oof_top1_accuracy':float(oof_correct.mean()),
            'validation_correctness_brier':float(np.mean((confidence-top_correct)**2)),
            'validation':selective_metrics(va,votes,model.classes_,records),
            'test':selective_metrics(ta,tp,model.classes_,test_records),
            'limitations':'Validation contains known dish classes only. This is not validated out-of-distribution detection and cannot identify unseen plain milk.'}
    report={'schema_version':3,'training_images':len(xs),'classes':len(model.classes_),
            'training_top1_accuracy':float(np.mean(model.predict(xs)==ys)),
            'validation_trials':trials,'selected':selected,'test':metrics(model,tx,test_records),
            'rejection':safety,'selection':'Training-only 3-fold correctness calibration; C and rejection threshold selected on validation; test evaluated after freezing. Test set has been reused across previous development rounds, so it is not a fresh blind benchmark.',
            'encoder':{'name':'MobileNetV2 ImageNet frozen encoder','sha256':ENCODER_SHA256,'dimensions':1280},
            'limitations':'Unknown classes such as plain milk are absent. All dish profiles require user confirmation.'}
    import shutil
    if MODEL.exists() and not (ROOT/'model_before_refinement.npz').exists():
        shutil.copyfile(MODEL,ROOT/'model_before_refinement.npz')
        if (ROOT/'training_report.json').exists():shutil.copyfile(ROOT/'training_report.json',ROOT/'training_report_before_refinement.json')
    temporary=ROOT/'model.pending.npz'
    np.savez_compressed(temporary,x=xs,y=ys,kind=np.array('deep_svc'),c=np.array(selected['C']),version=np.array(2),feature_kind=np.array(feature_kind),encoder_sha256=np.array(ENCODER_SHA256),
                        probability_threshold=np.array(threshold),distance_limit=np.array(distance_limit),safety_version=np.array(2),
                        risk_coefficients=coefficients,risk_intercept=np.array(intercept))
    temporary.replace(MODEL)
    (ROOT/'training_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def number(v):
    try:
        n = float(v)
        return round(n, 3) if math.isfinite(n) and n >= 0 else None
    except (TypeError, ValueError):
        return None


def rating(n):
    """Transparent prototype heuristic; deliberately NOT named Nutri-Score."""
    needed = ['energy_kcal','saturated_fat_g','sodium_mg','free_sugar_g','fibre_g','protein_g']
    if any(n.get(k) is None for k in needed):
        return {'grade': None, 'system': 'Prototype A–F', 'reason': 'Missing inputs; no grade calculated.'}
    penalty = n['energy_kcal']/100 + n['saturated_fat_g']/2 + n['sodium_mg']/200 + n['free_sugar_g']/5
    benefit = min(n['fibre_g']/2, 5) + min(n['protein_g']/5, 3)
    score = round(penalty-benefit, 2)
    grade = 'ABCDEF'[sum(score > t for t in (0, 2, 4, 6, 9))]
    return {'grade': grade, 'system': 'Prototype A–F (not a standard or medical assessment)', 'score': score}


PROFILE_KEYS = ('energy_kcal', 'protein_g', 'carbohydrate_g', 'fat_g', 'saturated_fat_g',
                'free_sugar_g', 'fibre_g', 'sodium_mg')


def ingredient_tokens(value):
    # Split only commas outside parentheses; preserve alternative/local ingredient names.
    parts = re.split(r',\s*(?![^()]*\))', value)
    return list(dict.fromkeys(p.strip() for p in parts if p.strip()))


def ingredient_profile(items):
    if not isinstance(items, list) or not 1 <= len(items) <= 100:
        raise ValueError('Provide 1–100 ingredient rows.')
    rows = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Each ingredient must be an object.')
        grams = number(item.get('grams'))
        if grams is None or grams <= 0 or grams > 100000:
            raise ValueError('Every included ingredient needs a weight greater than 0 and at most 100000 g.')
        raw = item.get('nutrition_per_100g', {})
        if not isinstance(raw, dict):
            raise ValueError('Ingredient nutrition must be an object.')
        nutrition = {}
        for key in PROFILE_KEYS:
            value = raw.get(key)
            nutrition[key] = None if value is None or value == '' else number(value)
            if value is not None and value != '' and nutrition[key] is None:
                raise ValueError('Nutrition values must be finite, nonnegative numbers.')
        rows.append({'name':str(item.get('name','Ingredient'))[:150], 'grams':grams,
                     'nutrition_per_100g':nutrition,
                     'contribution':{k:None if v is None else round(v*grams/100,3) for k,v in nutrition.items()},
                     'source':str(item.get('source','User-entered estimate'))[:200]})
    weight = sum(r['grams'] for r in rows)
    total = {k:round(sum(r['contribution'][k] for r in rows),3)
             if all(r['contribution'][k] is not None for r in rows) else None for k in PROFILE_KEYS}
    per100 = {k:None if v is None else round(v*100/weight,3) for k,v in total.items()}
    return {'ingredients':rows, 'total_weight_g':round(weight,3), 'total_nutrition':total,
            'nutrition_per_100g':per100, 'rating':rating(per100),
            'missing_inputs':{k:[r['name'] for r in rows if r['contribution'][k] is None] for k in PROFILE_KEYS if total[k] is None},
            'basis':'User-confirmed ingredient weights and per-100-g values; not a measured photo profile. Weight is sum of inputs, without cooking yield adjustment.'}


class Engine:
    def __init__(self):
        wb = openpyxl.load_workbook(DATA / 'Anuvaad_INDB_2024.11.xlsx', read_only=True, data_only=True)
        rows = iter(wb.active.values)
        headers = next(rows)
        self.foods = [dict(zip(headers, r)) for r in rows if r[1]]
        wb.close()
        self.names = [normalize(r['food_name']) for r in self.foods]
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
        self.vectors = self.vectorizer.fit_transform(self.names)
        recipe_path = DATA / 'Cleaned_Indian_Food_Dataset.csv'
        self.recipes = []
        if recipe_path.exists():
            with recipe_path.open(encoding='utf-8-sig', newline='') as f:
                self.recipes = [r for r in csv.DictReader(f) if r.get('TranslatedRecipeName') and r.get('Cleaned-Ingredients')]
        self.recipe_names = [normalize(r['TranslatedRecipeName']) for r in self.recipes]
        self.recipe_vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=['recipe', 'style', 'recipes']) if self.recipes else None
        self.recipe_vectors = self.recipe_vectorizer.fit_transform(self.recipe_names) if self.recipes else None
        self.model = None
        self.raw_classes = set()
        if MODEL.exists():
            with np.load(MODEL, allow_pickle=False) as d:
                self.feature_kind = str(d['feature_kind']) if 'feature_kind' in d else 'handcrafted'
                if self.feature_kind == 'mobilenetv2' and str(d['encoder_sha256']) != ENCODER_SHA256:
                    raise ValueError('Image encoder does not match the training artifact.')
                self.model_version = int(d['version']) if 'version' in d else 1
                self.model = make_image_model(str(d['kind']), float(d['c']),raw_classes=tuple(str(v) for v in d['raw_classes']) if 'raw_classes' in d else (),routing_threshold=float(d['routing_threshold']) if 'routing_threshold' in d else .5) if self.model_version == 2 else make_image_model('extra_trees')
                self.model.fit(d['x'], d['y'])
                self.raw_classes = set(str(x) for x in d['raw_classes']) if 'raw_classes' in d else set()
                self.dataset_revision = str(d['source_revision']) if 'source_revision' in d else None
                self.split_manifest_sha256 = str(d['split_manifest_sha256']) if 'split_manifest_sha256' in d else None
                self.probability_threshold = float(d['probability_threshold']) if 'probability_threshold' in d else 1.01
                self.distance_scaler, self.neighbors, computed_limit = novelty_model(d['x'],float(d['distance_limit']) if 'distance_limit' in d else None)
                self.distance_limit = float(d['distance_limit']) if 'distance_limit' in d else computed_limit
                self.probability_calibrated = 'risk_coefficients' in d
                self.risk_coefficients = d['risk_coefficients'].copy() if 'risk_coefficients' in d else np.zeros(3)
                self.risk_intercept = float(d['risk_intercept']) if 'risk_intercept' in d else -40

    def namesearch(self, query, limit=5):
        q = normalize(query)
        if not q:
            return []
        scores = (self.vectors @ self.vectorizer.transform([q]).T).toarray().ravel()
        results = []
        for i in np.argsort(-scores)[:limit]:
            if scores[i] < .22:
                continue
            if not any(SequenceMatcher(None, a, b).ratio() >= .6
                       for a in q.split() for b in self.names[i].split()):
                continue
            row = self.foods[i]
            n = {key: number(row.get(col)) for key,col in {
                'energy_kcal':'energy_kcal','energy_kj':'energy_kj','carbohydrate_g':'carb_g',
                'protein_g':'protein_g','fat_g':'fat_g','free_sugar_g':'freesugar_g',
                'fibre_g':'fibre_g','sodium_mg':'sodium_mg'}.items()}
            sat = number(row.get('sfa_mg'))
            n['saturated_fat_g'] = None if sat is None else round(sat/1000,3)
            results.append({'name':row['food_name'], 'food_code':row['food_code'], 'match_similarity':round(float(scores[i]),3), 'source':'Anuvaad INDB 2024.11', 'basis':'per 100 g (workbook basis; verify against dataset documentation)', 'nutrition':n, 'rating':rating(n)})
        return results

    def raw_ingredient_suggestion(self, label, allow_untrained=False):
        canonical = normalize(label)
        classes = {normalize(x):x for x in getattr(self,'raw_classes',set())}
        if canonical not in classes and not allow_untrained:
            return None
        label=classes.get(canonical,label.strip())
        exact=next((r for r in self.foods if normalize(str(r['food_name']).split('(')[0])==canonical),None)
        matches=[]
        if exact:
            matches=[r for r in self.namesearch(str(exact['food_name']),5) if r['food_code']==exact['food_code']]
        nutrition=matches[0]['nutrition'] if matches else {}
        item={'name':label,'grams':None,'nutrition_per_100g':{k:nutrition.get(k) for k in PROFILE_KEYS},
              'source':matches[0]['source'] if matches else 'No exact raw-ingredient nutrition record; supply verified per-100-g values',
              'nutrition_record':matches[0]['name'] if matches else None}
        return {'name':label,'inference':'Confirm the ingredient, food variant and preparation. Quantities and nutrition require a separate reference.',
                'ingredient_quantities_text':'Enter measured grams; do not infer quantity from the photo.',
                'ingredients':[item],'nutrition_matches':matches}

    def recipe_search(self, query, limit=3):
        q = normalize(query)
        if not q or not self.recipes:
            return []
        scores = (self.recipe_vectors @ self.recipe_vectorizer.transform([q]).T).toarray().ravel()
        results = []
        for i in np.argsort(-scores):
            if scores[i] < .18 or len(results) >= limit:
                break
            # Require query words to occur in title: avoid recipes with merely shared ingredients.
            meaningful = [w for w in q.split() if w not in ('recipe','recipes','style','and','with')]
            if not meaningful or not all(w in self.recipe_names[i].split() for w in meaningful):
                continue
            r = self.recipes[i]
            ingredients = []
            for name in ingredient_tokens(r['Cleaned-Ingredients']):
                # Never use fuzzy dish matches (e.g. onion soup) as a raw ingredient profile.
                exact = next((f for f, n in zip(self.foods,self.names)
                              if normalize(name) == normalize(str(f['food_name']).split('(')[0])), None)
                match = next((f for f in self.namesearch(str(exact['food_name']),5)
                              if f['food_code'] == exact['food_code']), None) if exact else None
                ingredients.append({'name':name, 'grams':None,
                                    'nutrition_per_100g':{k:match['nutrition'].get(k) if match else None for k in PROFILE_KEYS},
                                    'source':match['source'] if match else 'Unavailable in local nutrition workbook',
                                    'nutrition_record':match['name'] if match else None})
            results.append({'recipe_id':int(i), 'name':r['TranslatedRecipeName'],
                            'match_similarity':round(float(scores[i]),3), 'source_url':r.get('URL',''),
                            'ingredient_quantities_text':r.get('TranslatedIngredients',''),
                            'ingredients':ingredients, 'inference':'Recipe-derived ingredient suggestions; not visible ingredient detection or segmentation.'})
        return results

    def analyze_image(self, im):
        if self.model is None:
            raise ValueError('Image model is not trained. Run: python fitness_food.py train')
        if self.model_version != 2 or not self.probability_calibrated:
            return {'status':'uncertain', 'candidates':[], 'reasons':['Run python fitness_food.py refine to enable calibrated recognition.'], 'message':'Food could not be identified reliably. Enter its name to continue.'}
        im = ImageOps.exif_transpose(im).convert('RGB')
        w,h=im.size
        xs=np.asarray([visual_features(im,self.feature_kind)])
        center=np.asarray([visual_features(im.crop((w*.15,h*.15,w*.85,h*.85)),self.feature_kind)])
        keep,probs,distances,stable,confidence=rejection_checks(self.model,xs,center,self.distance_scaler,self.neighbors,self.distance_limit,self.probability_threshold,self.risk_coefficients,self.risk_intercept)
        reasons=[]
        if confidence[0]<self.probability_threshold:reasons.append('Insufficient estimated recognition reliability.')
        if not stable[0]:reasons.append('Whole-photo and center-crop predictions disagree.')
        if distances[0]>self.distance_limit:reasons.append('Appearance differs from training examples.')
        candidates=[]
        if keep[0]:
            # Only offer a label for confirmation; never attach nutrition or ingredients here.
            candidates=[{'label':str(self.model.classes_[i]), 'rank':rank+1, 'kind':'raw ingredient' if str(self.model.classes_[i]) in getattr(self,'raw_classes',set()) else 'dish',
                         'requires_confirmation':True, 'nutrition_matches':[], 'recipe_matches':[]} for rank,i in enumerate(np.argsort(-probs[0])[:3])]
        return {'status':'needs_confirmation' if keep[0] else 'uncertain', 'candidates':candidates,
                'reasons':reasons, 'message':'Confirm the food name before viewing its profile.' if keep[0] else 'Food could not be identified reliably. Try a clearer photo or enter the food name; unfamiliar foods and cooking styles may be unsupported.',
                'known_classes':len(self.model.classes_), 'estimated_top1_correctness':round(float(confidence[0]),4), 'probability_note':'Reliability is estimated on trained foods only; unfamiliar ingredients and cooking styles can still resemble a known food.'}

    def classify(self, im):
        return self.analyze_image(im)['candidates']

    def evaluate(self, split):
        if getattr(self,'raw_classes',set()) and (DATA/'combined_split_manifest.json').exists():
            from train_combined import PLAN, encode_plan, domain_report
            plan=json.loads(PLAN.read_text())
            if (self.split_manifest_sha256 and hashlib.sha256(PLAN.read_bytes()).hexdigest()!=self.split_manifest_sha256) or plan['source_revision'] != self.dataset_revision:
                raise ValueError('Evaluation split manifest does not match the trained dataset revision.')
            all_x,all_center=encode_plan(plan)
            ids=np.array([i for i,r in enumerate(plan['records']) if r['split']==split])
            rows=[plan['records'][i] for i in ids]
            x=all_x[ids];center=all_center[ids]
            accepted,votes,_,_,_=rejection_checks(self.model,x,center,self.distance_scaler,self.neighbors,self.distance_limit,self.probability_threshold,self.risk_coefficients,self.risk_intercept)
            return {'split':split,'domains':domain_report(self.model,x,rows,accepted,votes),'classes':len(self.model.classes_)}

        if self.model_version == 2:
            xs, _, records = split_features(split,self.feature_kind)
            return {'split':split, **metrics(self.model, xs, records), 'classes':len(self.model.classes_)}
        correct = total = 0
        for filename, _, labels in image_records(split):
            with Image.open(IMAGES / split / filename) as im:
                w,h=im.size
                views=[im, im.crop((w*.1,h*.1,w*.9,h*.9)),im.crop((w*.2,h*.2,w*.8,h*.8))]
                predicted=self.model.classes_[self.model.predict_proba([features(v) for v in views]).mean(axis=0).argmax()]
            correct += predicted in labels
            total += 1
        return {'split':split, 'images':total, 'top1_accuracy':correct/total if total else None}


def open_food_facts_lookup(code):
    if not re.fullmatch(r'\d{8}|\d{12,14}', code):
        raise ValueError('Use an 8, 12, 13 or 14 digit product barcode.')
    url = f'https://world.openfoodfacts.org/api/v2/product/{code}.json'
    req = urllib.request.Request(url, headers={'User-Agent':'FitnessJournalLocal/0.1 (local educational prototype)'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError) as e:
        raise ValueError('Open Food Facts is unavailable. Check your internet connection.') from e
    if data.get('status') != 1:
        return {'barcode':code,'found':False,'source_url':url}
    p = data['product']; raw = p.get('nutriments', {})
    n = {key:number(raw.get(col)) for key,col in {'energy_kcal':'energy-kcal_100g','energy_kj':'energy_100g','protein_g':'proteins_100g','carbohydrate_g':'carbohydrates_100g','fat_g':'fat_100g','saturated_fat_g':'saturated-fat_100g','total_sugar_g':'sugars_100g','fibre_g':'fiber_100g'}.items()}
    sodium = number(raw.get('sodium_100g'))
    n['sodium_mg'] = None if sodium is None else round(sodium*1000,3)
    grade = str(p.get('nutrition_grades',p.get('nutriscore_grade',''))).upper()
    return {'barcode':code,'found':True,'name':p.get('product_name') or 'Unnamed product','source':'Open Food Facts (ODbL)','source_url':url,'basis':'per 100 g / 100 ml, as reported on the product','nutrition':n,'rating':{'grade':grade if grade in list('ABCDE') else None,'system':'Nutri-Score A–E, supplied by Open Food Facts'}}


def barcode_lookup(code):
    if not re.fullmatch(r'\d{8}|\d{12,14}',code):
        raise ValueError('Use an 8, 12, 13 or 14 digit product barcode.')
    warnings=[];product=None
    try:product=services.barcode_product(code)
    except services.ProviderError as error:warnings.append(str(error))
    try:nutrition=open_food_facts_lookup(code)
    except ValueError as error:
        warnings.append(str(error));nutrition={'barcode':code,'found':False}
    if product and product.get('found'):
        if nutrition.get('found'):
            result={**nutrition,'name':product['name'],'brand':product.get('brand',''),
                    'source':'Barcode Lookup + Open Food Facts','source_url':product['source_url'],
                    'nutrition_source_url':nutrition['source_url']}
        else:
            result={**product,'basis':'Nutrition unavailable; no matching Open Food Facts profile.',
                    'nutrition':{k:None for k in (*PROFILE_KEYS,'energy_kj','total_sugar_g')},
                    'rating':{'grade':None,'system':'Nutri-Score A–E','reason':'Barcode Lookup identifies the product but does not supply this nutrition profile.'}}
    else:result=nutrition
    if warnings:result['warnings']=warnings
    return result


def search_food(engine,query='',encoded_image=None):
    query=str(query).strip()[:300]
    if not query and not encoded_image:raise ValueError('Enter a food name, barcode, or upload an image.')
    result={'name_matches':[], 'image_candidates':[], 'barcodes':[], 'warnings':[], 'recipe_matches':[]}
    codes=[];image=None
    if query.isdigit():codes.append(query)
    elif query:
        result['name_matches']=engine.namesearch(query)
        result['recipe_matches']=engine.recipe_search(query)
        ingredient=engine.raw_ingredient_suggestion(query)
        if ingredient:
            result['ingredient_suggestion']=ingredient;result['name_matches']=ingredient['nutrition_matches'];result['recipe_matches']=[]
    if encoded_image:
        import zxingcpp
        raw=base64.b64decode(encoded_image,validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load();image=ImageOps.exif_transpose(opened).convert('RGB')
        codes.extend(r.text for r in zxingcpp.read_barcodes(image) if r.text.isdigit())
        if not codes:
            try:
                with INFER_LOCK:analysis=engine.analyze_image(image)
            except ValueError:
                analysis={'status':'uncertain','candidates':[],'message':'Local image recognition is unavailable.','reasons':['Local model could not identify the image.']}
            result['image_analysis']=analysis;result['image_candidates']=analysis['candidates']
            result['warnings'].append('Confirm any image suggestion before viewing nutrition; portions cannot be inferred from a photo.')
    if codes:
        for code in dict.fromkeys(codes):
            try:
                record=barcode_lookup(code);result['barcodes'].append(record)
                result['warnings'].extend(record.get('warnings',[]))
            except ValueError as error:result['warnings'].append(str(error))
        return result  # A barcode is never sent to Gemini as a food name or image.
    unrecognized=(result.get('image_analysis',{}).get('status')=='uncertain') if image is not None else not (result['name_matches'] or result['recipe_matches'] or result.get('ingredient_suggestion'))
    if unrecognized:
        fallback=services.closest_food(query,image)
        result['gemini_fallback']=fallback
        if image is not None and fallback['candidates']:
            local=result['image_analysis']
            result['image_analysis']={**local,'status':'needs_confirmation','source':'Google Gemini',
                'message':fallback['message'],'candidates':fallback['candidates'],
                'probability_note':'Cloud suggestions are estimates; confirm the food identity. Nutrition is looked up separately.'}
            result['image_candidates']=fallback['candidates']
        elif not fallback['candidates']:result['warnings'].append(fallback['message'])
    return result


def confirmed_nutrition(engine,label,is_raw,ingredient,matches):
    """Resolve the confirmed identity; a fuzzy recipe is not its nutrition profile."""
    query=normalize(label)
    def exact(name):
        names=[str(name).split('(')[0],*re.findall(r'\(([^()]*)\)',str(name))]
        return any(normalize(n)==query for n in names)
    profile=next((r for r in matches if exact(r.get('name','')) and isinstance(r.get('nutrition'),dict)),None)
    if profile is None:
        for row in getattr(engine,'foods',[]):
            if exact(row['food_name']):
                profile=next((r for r in engine.namesearch(row['food_name']) if r.get('food_code')==row.get('food_code') and isinstance(r.get('nutrition'),dict)),None)
                if profile:break
    warnings=[]
    if profile is None or any(profile['nutrition'].get(k) is None for k in ('energy_kcal','protein_g','carbohydrate_g','fat_g')):
        estimate=services.estimate_nutrition(label,'raw ingredient' if is_raw else 'dish')
        if estimate.get('status')=='estimated':
            profile={**estimate,'rating':rating(estimate['nutrition'])}
            profile['rating']['system']+=' · computed from estimated values'
        else:
            warnings.append(estimate.get('message','Nutrition is unavailable.'))
    if profile and ingredient and profile.get('is_estimate') and profile.get('basis_unit')=='ml':
        ingredient['inference']=ingredient.get('inference','')+' The nutrition card is per 100 ml. The ingredient calculator uses grams and needs a density conversion before those values can be used.'
    if profile and ingredient and not (profile.get('is_estimate') and profile.get('basis_unit')=='ml'):
        for item in ingredient['ingredients']:
            item['nutrition_per_100g']={k:profile['nutrition'].get(k) for k in PROFILE_KEYS}
            item['source']=profile['source'];item['nutrition_record']=profile['name']
            if profile.get('is_estimate'):item['source']+=' (estimated per 100 g)'
    return profile,ingredient,warnings


ENGINE = None
INFER_LOCK = threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        raw = json.dumps(data, allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/':
            raw = (ROOT / 'index.html').read_bytes()
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(raw)
        elif self.path == '/api/status':
            config=services.configuration()
            self.send_json(200, {'model_ready': ENGINE.model is not None, 'foods':len(ENGINE.foods), 'recipes':len(ENGINE.recipes), 'recognition_refined':bool(getattr(ENGINE,'probability_calibrated',False)), 'image_classes':len(ENGINE.model.classes_) if ENGINE.model else 0, 'raw_ingredient_classes':len(ENGINE.raw_classes), 'gemini_configured':bool(config['gemini_key']), 'barcode_lookup_configured':bool(config['barcode_key'])})
        else:
            self.send_json(404, {'error':'Not found'})

    def do_POST(self):
        if self.path not in ('/api/search', '/api/ingredients/profile', '/api/confirm-food'):
            return self.send_json(404, {'error':'Not found'})
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 12_000_000:
                raise ValueError('Request must be under 12 MB.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Request must be a JSON object.')
            if self.path == '/api/ingredients/profile':
                return self.send_json(200, ingredient_profile(body.get('ingredients')))
            if self.path == '/api/confirm-food':
                label = str(body.get('label', '')).strip()[:150]
                token=body.get('confirmation_token')
                gemini_kind=services.verify_suggestion(token,label) if token else None
                if token and not gemini_kind:raise ValueError('This Gemini suggestion expired or is invalid. Search again.')
                known=ENGINE.model is not None and label in ENGINE.model.classes_
                if not label or (not known and not gemini_kind):
                    raise ValueError('Choose one of the known food labels or a current Gemini suggestion, or search by name.')
                is_raw=gemini_kind=='raw ingredient' if gemini_kind else label in getattr(ENGINE,'raw_classes',set())
                if gemini_kind=='raw ingredient':ingredient=ENGINE.raw_ingredient_suggestion(label,allow_untrained=True)
                else:ingredient=ENGINE.raw_ingredient_suggestion(label) if is_raw else None
                matches=ingredient['nutrition_matches'] if ingredient else ENGINE.namesearch(label)
                profile,ingredient,warnings=confirmed_nutrition(ENGINE,label,is_raw,ingredient,matches)
                return self.send_json(200, {'name_matches':matches, 'nutrition_profile':profile,'warnings':warnings,
                    'recipe_matches':[] if ingredient else ENGINE.recipe_search(label), 'ingredient_suggestion':ingredient, 'confirmed_label':label,
                    'recognition_source':'Google Gemini' if gemini_kind else 'Local model'})
            query = str(body.get('query','')).strip()[:300]
            self.send_json(200,search_food(ENGINE,query,body.get('image')))
        except (ValueError, KeyError, OSError, Image.DecompressionBombError) as e:
            self.send_json(400, {'error':str(e)})
        except Exception:
            self.send_json(500, {'error':'Search failed. Check the server console.'})
            import traceback
            traceback.print_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','refine','serve','evaluate'],nargs='?',default='serve')
    p.add_argument('--port',type=int,default=8000)
    args = p.parse_args()
    if args.command == 'train':
        train(); return
    if args.command == 'refine':
        refine(); return
    global ENGINE
    ENGINE = Engine()
    if args.command == 'evaluate':
        if ENGINE.model is None:
            p.error('Train the model first.')
        print(json.dumps(ENGINE.evaluate('test'),indent=2)); return
    print(f'Open http://127.0.0.1:{args.port} — model ready: {ENGINE.model is not None}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__ == '__main__':
    main()
