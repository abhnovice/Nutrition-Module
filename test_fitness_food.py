import io
import json
import unittest
from unittest.mock import patch
from PIL import Image
import fitness_food as app

def lookup_only_engine():
    # These unit cases exercise lookups; real checkpoint loading is checked separately.
    with patch.object(app,'MODEL') as artifact:
        artifact.exists.return_value=False
        return app.Engine()

class FoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = lookup_only_engine()

    def test_workbook_lookup_and_units(self):
        food = self.engine.namesearch('Hot tea (Garam Chai)')[0]
        self.assertEqual(food['food_code'], 'ASC001')
        self.assertAlmostEqual(food['nutrition']['saturated_fat_g'], .321, places=3)
        self.assertGreater(food['nutrition']['energy_kcal'], 16)
        self.assertEqual(self.engine.namesearch('zzzzzxqvqqq'), [])

    def test_missing_rating_not_invented(self):
        self.assertIsNone(app.rating({})['grade'])
        self.assertIsNone(app.number(float('nan')))

    def test_barcode_units_and_official_grade(self):
        payload={'status':1,'product':{'product_name':'Test','nutrition_grades':'b','nutriments':{'sodium_100g':.4,'energy-kcal_100g':125}}}
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(payload).encode())), patch('external_services.barcode_product',return_value=None):
            result=app.barcode_lookup('12345678')
        self.assertEqual(result['nutrition']['sodium_mg'],400)
        self.assertIsNone(result['nutrition']['protein_g'])
        self.assertEqual(result['rating']['grade'],'B')
        with self.assertRaises(ValueError):app.barcode_lookup('../bad')

    def test_generated_barcode_decodes(self):
        import zxingcpp
        barcode=zxingcpp.create_barcode('5901234123457', zxingcpp.BarcodeFormat.EAN13)
        im=barcode.to_image(scale=4,add_quiet_zones=True)
        decoded=zxingcpp.read_barcodes(im)
        self.assertEqual(decoded[0].text,'5901234123457')

    def test_features(self):
        vector=app.features(Image.new('RGB',(50,80),'red'))
        self.assertTrue(all(map(lambda x:x==x,vector)))
        self.assertEqual(vector.shape,(608,))


class ModelSelectionTests(unittest.TestCase):
    def test_image_annotations_grouped_and_dominant_area_used(self):
        rows = [dict(filename='a.jpg', **{'class':'small'}, xmin='0', ymin='0', xmax='5', ymax='5'),
                dict(filename='a.jpg', **{'class':'large'}, xmin='0', ymin='0', xmax='20', ymax='20')]
        with patch('fitness_food.samples', return_value=iter(rows)):
            records = app.image_records('train')
        self.assertEqual(records, [('a.jpg','large',{'small','large'})])

    def test_top_three_metric_and_multilabel_ground_truth(self):
        import numpy as np
        class Model:
            classes_ = np.array(['a','b','c','d'])
            def predict_proba(self, xs):
                return np.array([[.6,.2,.15,.05],[.6,.2,.15,.05]])
        result=app.metrics(Model(),[[0],[0]],[('1','b',{'b'}),('2','d',{'d','a'})])
        self.assertEqual(result['top1_accuracy'],.5)
        self.assertEqual(result['top3_accuracy'],1)

    def test_svc_votes_are_finite_and_rank_predictions(self):
        import numpy as np
        model=app.make_image_model('svc',1)
        x=np.array([[0,0],[0,.1],[1,1],[1,1.1],[2,2],[2,2.1]])
        model.fit(x,np.array(['a','a','b','b','c','c']))
        votes=app.model_votes(model,x)
        np.testing.assert_allclose(votes.sum(axis=1),1)
        np.testing.assert_array_equal(model.classes_[votes.argmax(axis=1)],model.predict(x))



class IngredientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = lookup_only_engine()

    def test_recipe_matches_and_no_fabricated_raw_nutrition(self):
        recipes=self.engine.recipe_search('palak paneer')
        self.assertTrue(recipes)
        self.assertIn('not visible ingredient',recipes[0]['inference'])
        ingredients=[i for r in recipes for i in r['ingredients']]
        onion=next(i for i in ingredients if i['name']=='onion')
        self.assertIsNone(onion['nutrition_record'])
        self.assertTrue(all(v is None for v in onion['nutrition_per_100g'].values()))
        self.assertEqual(self.engine.recipe_search('zzzzqxq'),[])

    def test_ingredient_split_preserves_parentheses_and_deduplicates(self):
        self.assertEqual(app.ingredient_tokens('salt, oil (olive, sunflower),salt'),['salt','oil (olive, sunflower)'])

    def test_weighted_profile_and_missing_values(self):
        profile=app.ingredient_profile([
            {'name':'first','grams':50,'nutrition_per_100g':{'energy_kcal':100,'protein_g':10}},
            {'name':'second','grams':150,'nutrition_per_100g':{'energy_kcal':200,'protein_g':20}}])
        self.assertEqual(profile['total_weight_g'],200)
        self.assertEqual(profile['total_nutrition']['energy_kcal'],350)
        self.assertEqual(profile['nutrition_per_100g']['energy_kcal'],175)
        self.assertEqual(profile['ingredients'][0]['contribution']['protein_g'],5)
        self.assertIsNone(profile['total_nutrition']['sodium_mg'])
        self.assertIsNone(profile['rating']['grade'])
        self.assertEqual(profile['missing_inputs']['sodium_mg'],['first','second'])

    def test_complete_profile_rating_and_unknown_not_zero(self):
        nutrients={k:0 for k in app.PROFILE_KEYS}
        profile=app.ingredient_profile([{'name':'known','grams':100,'nutrition_per_100g':nutrients}])
        self.assertEqual(profile['rating']['grade'],'A')
        profile=app.ingredient_profile([{'name':'known','grams':100,'nutrition_per_100g':nutrients},
                                       {'name':'unknown','grams':100}])
        self.assertIsNone(profile['total_nutrition']['energy_kcal'])
        self.assertIsNone(profile['rating']['grade'])

    def test_invalid_weight_or_nutrition_rejected(self):
        for grams in [None,0,-1,'NaN',100001]:
            with self.assertRaises(ValueError):app.ingredient_profile([{'grams':grams}])
        with self.assertRaises(ValueError):
            app.ingredient_profile([{'grams':100,'nutrition_per_100g':{'protein_g':-1}}])
        with self.assertRaises(ValueError):app.ingredient_profile([])

    def test_profile_and_recipe_search_api(self):
        for path, body in [('/api/ingredients/profile',{'ingredients':[{'name':'rice','grams':100,'nutrition_per_100g':{'energy_kcal':130}}]}),
                           ('/api/search',{'query':'palak paneer'})]:
            handler=app.Handler.__new__(app.Handler)
            raw=json.dumps(body).encode()
            handler.path=path;handler.headers={'Content-Length':str(len(raw))};handler.rfile=io.BytesIO(raw)
            captured=[];handler.send_json=lambda status,data:captured.append((status,data))
            with patch.object(app,'ENGINE',self.engine):handler.do_POST()
            self.assertEqual(captured[0][0],200)
            if path=='/api/search':self.assertTrue(captured[0][1]['recipe_matches'])
            else:self.assertEqual(captured[0][1]['total_nutrition']['energy_kcal'],130)



class RejectionTests(unittest.TestCase):
    def test_high_rank_score_is_not_a_correctness_probability(self):
        import numpy as np
        votes=np.array([[.99,.01]])
        estimate=app.correctness_estimate(votes,np.zeros(3),-4)
        self.assertLess(estimate[0],.02)

    def test_rejection_for_instability_distance_and_low_reliability(self):
        import numpy as np
        class Scaler:
            def transform(self,x):return x
        class Neighbors:
            def kneighbors(self,x,n_neighbors):return np.array([[5.]]),np.array([[0]])
        class Model:
            def __init__(self):self.calls=0
            def predict_proba(self,x):
                self.calls+=1
                return np.array([[.9,.1]]) if self.calls==1 else np.array([[.1,.9]])
        keep,_,distance,stable,confidence=app.rejection_checks(Model(),np.zeros((1,2)),np.zeros((1,2)),Scaler(),Neighbors(),1,.7,np.zeros(3),-4)
        self.assertFalse(keep[0]);self.assertFalse(stable[0]);self.assertGreater(distance[0],1)
        self.assertLess(confidence[0],.7)

    def test_rejected_image_has_no_dish_or_ingredient_profiles(self):
        import numpy as np
        from types import SimpleNamespace
        engine=app.Engine.__new__(app.Engine)
        engine.model=SimpleNamespace(classes_=np.array(['doodhpak','kheer']))
        engine.model_version=2;engine.feature_kind='handcrafted';engine.probability_calibrated=True
        engine.probability_threshold=.7;engine.distance_limit=1;engine.distance_scaler=None;engine.neighbors=None
        engine.risk_coefficients=np.zeros(3);engine.risk_intercept=0
        fake=(np.array([False]),np.array([[.99,.01]]),np.array([2.]),np.array([False]),np.array([.2]))
        with patch('fitness_food.rejection_checks',return_value=fake):
            result=engine.analyze_image(Image.new('RGB',(80,80),'white'))
        self.assertEqual(result['status'],'uncertain');self.assertEqual(result['candidates'],[])
        self.assertGreater(len(result['reasons']),0)

    def test_accepted_candidates_require_confirmation_and_have_no_profiles(self):
        import numpy as np
        from types import SimpleNamespace
        engine=app.Engine.__new__(app.Engine)
        engine.model=SimpleNamespace(classes_=np.array(['doodhpak','kheer']))
        engine.model_version=2;engine.feature_kind='handcrafted';engine.probability_calibrated=True
        engine.probability_threshold=.7;engine.distance_limit=1;engine.distance_scaler=None;engine.neighbors=None
        engine.risk_coefficients=np.zeros(3);engine.risk_intercept=0
        fake=(np.array([True]),np.array([[.99,.01]]),np.array([.5]),np.array([True]),np.array([.9]))
        with patch('fitness_food.rejection_checks',return_value=fake):
            result=engine.analyze_image(Image.new('RGB',(80,80),'white'))
        self.assertEqual(result['status'],'needs_confirmation')
        self.assertTrue(result['candidates'][0]['requires_confirmation'])
        self.assertEqual(result['candidates'][0]['nutrition_matches'],[])
        self.assertEqual(result['candidates'][0]['recipe_matches'],[])
        self.assertNotIn('probability',result['candidates'][0])

    def test_confirmation_endpoint_validates_labels(self):
        import numpy as np
        from types import SimpleNamespace
        engine=SimpleNamespace(model=SimpleNamespace(classes_=np.array(['doodhpak'])),namesearch=lambda label:[{'name':label}],recipe_search=lambda label:[])
        for label,expected in [('doodhpak',200),('milk',400)]:
            handler=app.Handler.__new__(app.Handler);handler.path='/api/confirm-food'
            raw=json.dumps({'label':label}).encode();handler.headers={'Content-Length':str(len(raw))};handler.rfile=io.BytesIO(raw)
            captured=[];handler.send_json=lambda status,data:captured.append((status,data))
            with patch.object(app,'ENGINE',engine),patch.object(app.services,'estimate_nutrition',return_value={'status':'unavailable','message':'Estimate not requested by this routing test.'}):handler.do_POST()
            self.assertEqual(captured[0][0],expected)



class RawFoodTests(unittest.TestCase):
    def test_hierarchical_scores_stay_finite_at_router_extremes(self):
        import numpy as np
        class Router:
            def predict_proba(self,x):return np.array([[0,1],[1,0]],dtype=np.float32)
        class Head:
            def __init__(self,classes):self.classes_=np.array(classes)
            def decision_function(self,x):return np.array([1,-1])
        model=app.HierarchicalFoodClassifier()
        model.classes_=np.array(['dish a','dish b','raw a','raw b'])
        model.router_=Router();model.raw_model_=Head(['raw a','raw b']);model.dish_model_=Head(['dish a','dish b'])
        scores=model.decision_function(np.zeros((2,2)))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(model.classes_[scores[0].argmax()],'raw b')
        self.assertEqual(model.classes_[scores[1].argmax()],'dish a')

    def test_label_normalization_keeps_milk_variants_distinct(self):
        from prepare_raw_food import canonical_label
        self.assertEqual(canonical_label('vanilla_yoghurt'),'vanilla yogurt')
        self.assertEqual(canonical_label('soy_milk'),'soy milk')
        self.assertNotEqual(canonical_label('soy_milk'),canonical_label('milk'))

    def test_split_plan_is_content_grouped_and_reproducible(self):
        import hashlib
        from train_combined import plan_records
        rows=[]
        for label in ['milk','tomato']:
            for split,count in [('train',7),('validation',6)]:
                for i in range(count):
                    key=f'{label}-{split}-{i}'
                    digest=hashlib.sha256(key.encode()).hexdigest()
                    rows.append({'label':label,'path':key+'.jpg','source_split':split,'sha256':digest,'phash':digest[:16]})
        duplicate=dict(rows[0]);duplicate['source_split']='validation';duplicate['path']='duplicate.jpg';rows.append(duplicate)
        raw={'repository':'test','revision':'fixed','original_splits':{},'records':rows}
        with patch('fitness_food.image_records',return_value=[]):
            plan=plan_records(raw)
            reverse=plan_records({**raw,'records':list(reversed(rows))})
        self.assertEqual(plan['records'],reverse['records'])
        assigned={}
        for r in plan['records']:
            self.assertNotIn(r['sha256'],assigned)
            assigned[r['sha256']]=r['split']
        self.assertNotEqual(assigned[rows[0]['sha256']],'train')
        self.assertEqual({r['split'] for r in plan['records']},{'train','valid','test'})
        self.assertEqual(plan['source_exact_duplicates'],1)

    def test_rejection_target_must_hold_in_both_domains(self):
        import numpy as np
        from train_combined import combined_threshold
        confidence=np.full(60,.8);stable=np.ones(60,dtype=bool);distances=np.zeros(60)
        domains=np.array(['dish']*30+['raw']*30)
        correct=np.array([True]*15+[False]*15+[True]*30)
        self.assertEqual(combined_threshold(confidence,stable,distances,1,correct,domains),1.01)
        correct=np.array(([True]*24+[False]*6)*2)
        self.assertEqual(combined_threshold(confidence,stable,distances,1,correct,domains),.5)

    def test_domain_macro_metric_does_not_hide_milk_errors(self):
        import numpy as np
        from train_combined import domain_report
        class Model:
            classes_=np.array(['milk','tomato','doodhpak'])
            def predict_proba(self,x):
                # Input is a test-controlled winning class index.
                return np.eye(3)[np.asarray(x)[:,0].astype(int)]
        rows=[{'path':'m','label':'milk','labels':['milk'],'domain':'raw'},
              *[{'path':str(i),'label':'tomato','labels':['tomato'],'domain':'raw'} for i in range(3)],
              {'path':'d','label':'doodhpak','labels':['doodhpak'],'domain':'dish'}]
        report=domain_report(Model(),np.array([[1],[1],[1],[1],[2]]),rows)
        self.assertEqual(report['raw']['raw']['top1_accuracy'],.75)
        self.assertEqual(report['raw']['raw']['macro_top1_accuracy'],.5)
        self.assertEqual(report['dish']['raw']['top1_accuracy'],1)

    def test_raw_milk_does_not_use_dessert_nutrition_or_recipe(self):
        engine=app.Engine.__new__(app.Engine)
        engine.raw_classes={'milk','tomato'}
        engine.foods=[{'food_name':'Milk pudding','food_code':'P1'}]
        engine.namesearch=lambda *args:[{'name':'Milk pudding','food_code':'P1'}]
        item=engine.raw_ingredient_suggestion('milk')
        self.assertEqual(item['nutrition_matches'],[])
        self.assertEqual(item['ingredients'][0]['name'],'milk')
        self.assertTrue(all(v is None for v in item['ingredients'][0]['nutrition_per_100g'].values()))
        self.assertIsNone(engine.raw_ingredient_suggestion('milk pudding'))

    def test_raw_confirmation_suppresses_recipe_suggestions(self):
        import numpy as np
        from types import SimpleNamespace
        ingredient={'name':'milk','ingredients':[],'nutrition_matches':[]}
        engine=SimpleNamespace(model=SimpleNamespace(classes_=np.array(['milk','doodhpak'])),raw_classes={'milk'},
            raw_ingredient_suggestion=lambda label:ingredient,
            namesearch=lambda label:(_ for _ in ()).throw(AssertionError('Do not fuzzy-match raw milk')),
            recipe_search=lambda label:(_ for _ in ()).throw(AssertionError('Do not infer a milk recipe')))
        handler=app.Handler.__new__(app.Handler);handler.path='/api/confirm-food'
        raw=json.dumps({'label':'milk'}).encode();handler.headers={'Content-Length':str(len(raw))};handler.rfile=io.BytesIO(raw)
        captured=[];handler.send_json=lambda status,data:captured.append((status,data))
        with patch.object(app,'ENGINE',engine),patch.object(app.services,'estimate_nutrition',return_value={'status':'unavailable','message':'Estimate not requested by this routing test.'}):handler.do_POST()
        self.assertEqual(captured[0][0],200)
        self.assertEqual(captured[0][1]['recipe_matches'],[])
        self.assertEqual(captured[0][1]['ingredient_suggestion']['name'],'milk')

if __name__=='__main__':unittest.main()
