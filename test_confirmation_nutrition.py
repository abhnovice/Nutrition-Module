import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import fitness_food as app
import external_services as services


def estimate(unit='g'):
    values={'energy_kcal':1.0,'energy_kj':4.184,'protein_g':.2,'carbohydrate_g':0,'fat_g':0,'saturated_fat_g':0,'free_sugar_g':0,'total_sugar_g':0,'fibre_g':0,'sodium_mg':1}
    return {'status':'estimated','name':'green tea','is_estimate':True,'source':'Google Gemini · generic estimate','basis':'Estimated per 100 '+unit,'basis_unit':unit,'nutrition':values,'preparation':'Brewed unsweetened tea','assumptions':'No milk or sugar.'}


class ConfirmationNutritionTests(unittest.TestCase):
    def test_exact_local_profile_is_preferred_without_cloud_call(self):
        profile={'name':'Spinach paneer (Palak paneer)','source':'INDB','nutrition':estimate()['nutrition'],'rating':{'grade':'A'}}
        engine=SimpleNamespace(foods=[])
        with patch.object(services,'estimate_nutrition') as cloud:
            found,_,warnings=app.confirmed_nutrition(engine,'palak paneer',False,None,[profile])
        self.assertEqual(found,profile);self.assertEqual(warnings,[]);cloud.assert_not_called()

    def test_green_tea_never_uses_fuzzy_sweet_milk_tea_profile(self):
        chai={'name':'Hot tea (Garam Chai)','source':'INDB','nutrition':{**estimate()['nutrition'],'energy_kcal':50},'rating':{'grade':'C'}}
        with patch.object(services,'estimate_nutrition',return_value=estimate('ml')) as cloud:
            profile,_,_=app.confirmed_nutrition(SimpleNamespace(foods=[]),'green tea',False,None,[chai])
        self.assertTrue(profile['is_estimate']);self.assertEqual(profile['basis_unit'],'ml')
        self.assertEqual(profile['nutrition']['carbohydrate_g'],0);cloud.assert_called_once_with('green tea','dish')

    def test_raw_ingredient_rows_are_populated_only_with_mass_basis(self):
        ingredient={'ingredients':[{'name':'ginger','nutrition_per_100g':{},'source':'missing'}]}
        with patch.object(services,'estimate_nutrition',return_value=estimate()):
            profile,filled,_=app.confirmed_nutrition(SimpleNamespace(foods=[]),'ginger',True,ingredient,[])
        self.assertEqual(filled['ingredients'][0]['nutrition_per_100g']['energy_kcal'],1)
        beverage={'ingredients':[{'name':'milk','nutrition_per_100g':{'energy_kcal':None},'source':'missing'}]}
        with patch.object(services,'estimate_nutrition',return_value=estimate('ml')):
            _,filled,_=app.confirmed_nutrition(SimpleNamespace(foods=[]),'milk',True,beverage,[])
        self.assertIsNone(filled['ingredients'][0]['nutrition_per_100g']['energy_kcal'])

    def test_outage_is_explicit_without_substituting_wrong_recipe(self):
        with patch.object(services,'estimate_nutrition',return_value={'status':'unavailable','message':'Quota reached.'}):
            profile,_,warnings=app.confirmed_nutrition(SimpleNamespace(foods=[]),'green tea',False,None,[])
        self.assertIsNone(profile);self.assertEqual(warnings,['Quota reached.'])

    def test_confirm_endpoint_returns_populated_nutrition_profile(self):
        with patch.object(app,'MODEL') as model:
            model.exists.return_value=False;engine=app.Engine()
        label='green tea';body=json.dumps({'label':label,'confirmation_token':services.suggestion_token(label,'dish')}).encode()
        handler=app.Handler.__new__(app.Handler);handler.path='/api/confirm-food';handler.headers={'Content-Length':str(len(body))};handler.rfile=io.BytesIO(body)
        captured=[];handler.send_json=lambda status,data:captured.append((status,data))
        with patch.object(app,'ENGINE',engine),patch.object(services,'estimate_nutrition',return_value=estimate('ml')):
            handler.do_POST()
        self.assertEqual(captured[0][0],200)
        self.assertEqual(captured[0][1]['nutrition_profile']['nutrition']['energy_kcal'],1)
        self.assertTrue(captured[0][1]['nutrition_profile']['is_estimate'])

    def test_estimator_units_zero_values_and_invalid_numbers(self):
        config={'gemini_key':'test-key','gemini_model':'test-model','barcode_key':''}
        values={k:0 for k in services.NUTRITION_KEYS};values['energy_kcal']=1
        def response(nutrients):
            data={'can_estimate':True,'basis':'100 ml','preparation':'Brewed','assumptions':'Unsweetened','nutrition':nutrients}
            return ('test-model',{'candidates':[{'content':{'parts':[{'text':json.dumps(data)}]}}]})
        with patch.object(services,'configuration',return_value=config),patch.object(services,'generate_json',return_value=response(values)):
            profile=services.estimate_nutrition('green tea','dish')
        self.assertEqual(profile['status'],'estimated');self.assertEqual(profile['basis_unit'],'ml')
        self.assertEqual(profile['nutrition']['energy_kj'],4.184);self.assertEqual(profile['nutrition']['free_sugar_g'],0)
        for invalid in [-1,True,float('nan'),1001]:
            with patch.object(services,'configuration',return_value=config),patch.object(services,'generate_json',return_value=response({**values,'energy_kcal':invalid})):
                self.assertEqual(services.estimate_nutrition('green tea','dish')['status'],'unavailable')

if __name__=='__main__':unittest.main()
