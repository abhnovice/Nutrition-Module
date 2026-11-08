import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
from PIL import Image
import numpy as np
import external_services as services
import fitness_food as app

CONFIG={'gemini_key':'dummy-gemini-key','gemini_model':'gemini-test-flash','barcode_key':'dummy-barcode-key'}


def picture():
    data=io.BytesIO();Image.new('RGB',(40,40),'white').save(data,format='PNG')
    return base64.b64encode(data.getvalue()).decode()


def engine(known=False):
    return SimpleNamespace(namesearch=lambda name:[],recipe_search=lambda name:[],raw_ingredient_suggestion=lambda name:None,
        analyze_image=lambda image:{'status':'needs_confirmation' if known else 'uncertain',
            'candidates':[{'label':'milk','requires_confirmation':True}] if known else [],'message':'local result'})


class CloudFallbackTests(unittest.TestCase):
    def test_gemini_multimodal_request_and_signed_confirmation(self):
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'is_food':True,'matches':[
            {'name':'Milk','kind':'raw ingredient','reason':'White liquid; a specific dessert is not established.'}]})}]}}]}
        with patch.object(services,'configuration',return_value=CONFIG),patch.object(services,'api_json',return_value=response) as request:
            result=services.closest_food('milk in a pan',Image.new('RGB',(2048,1000),'white'))
        self.assertEqual(result['status'],'suggestions')
        candidate=result['candidates'][0]
        self.assertEqual(services.verify_suggestion(candidate['confirmation_token'],'Milk'),'raw ingredient')
        self.assertEqual(candidate['nutrition_matches'],[])
        req=request.call_args.args[0];body=json.loads(req.data)
        self.assertNotIn(CONFIG['gemini_key'],req.full_url)
        self.assertEqual(req.get_header('X-goog-api-key'),CONFIG['gemini_key'])
        inline=body['contents'][0]['parts'][1]['inlineData']
        with Image.open(io.BytesIO(base64.b64decode(inline['data']))) as im:self.assertLessEqual(max(im.size),1024)
        self.assertEqual(inline['mimeType'],'image/jpeg')

    def test_missing_key_nonfood_and_quota_fail_gracefully(self):
        with patch.object(services,'configuration',return_value={**CONFIG,'gemini_key':''}),patch.object(services,'api_json') as request:
            self.assertEqual(services.closest_food('thing')['status'],'not_configured');request.assert_not_called()
        for reply in [{'candidates':[]},{'candidates':[{'content':{'parts':[{'text':'{"is_food": false, "matches": []}'}]}}]}, {'candidates':'invalid'}]:
            with patch.object(services,'configuration',return_value=CONFIG),patch.object(services,'api_json',return_value=reply):
                self.assertEqual(services.closest_food('thing')['candidates'],[])
        with patch.object(services,'configuration',return_value=CONFIG),patch.object(services,'api_json',side_effect=services.ProviderError('Gemini quota or rate limit reached.',429)):
            self.assertEqual(services.closest_food('thing')['status'],'unavailable')

    def test_transient_generation_error_uses_only_one_auto_alternative(self):
        import hashlib
        config={**CONFIG,'gemini_model':''}
        identity=hashlib.sha256(config['gemini_key'].encode()).hexdigest()
        response={'candidates':[{'content':{'parts':[{'text':'{"is_food":true,"matches":[{"name":"milk","kind":"raw ingredient","reason":"white liquid"}]}'}]}}]}
        with patch.object(services,'configuration',return_value=config),patch.object(services,'gemini_model',return_value='first-flash'),patch.dict(services._MODEL_OPTIONS,{identity:['first-flash','second-flash']},clear=True),patch.object(services,'api_json',side_effect=[services.ProviderError('Unavailable',503),response]) as call:
            result=services.closest_food('milk')
        self.assertEqual(result['status'],'suggestions');self.assertEqual(result['model'],'second-flash');self.assertEqual(call.call_count,2)
        with patch.object(services,'configuration',return_value=CONFIG),patch.object(services,'api_json',side_effect=services.ProviderError('Unavailable',503)) as call:
            self.assertEqual(services.closest_food('milk')['status'],'unavailable');self.assertEqual(call.call_count,1)

    def test_cloud_outage_preserves_local_name_results(self):
        local=engine();local.namesearch=lambda query:[{'name':'Local recipe'}]
        fallback={'status':'unavailable','candidates':[],'message':'Gemini quota or rate limit reached.'}
        with patch('zxingcpp.read_barcodes',return_value=[]),patch.object(app.services,'closest_food',return_value=fallback):
            result=app.search_food(local,query='dish',encoded_image=picture())
        self.assertEqual(result['name_matches'],[{'name':'Local recipe'}])
        self.assertIn(fallback['message'],result['warnings'])

    def test_tokens_reject_expiry_tampering_and_other_label(self):
        with patch.object(services.time,'time',return_value=100):token=services.suggestion_token('milk','raw ingredient')
        with patch.object(services.time,'time',return_value=101):
            self.assertEqual(services.verify_suggestion(token,'milk'),'raw ingredient')
            self.assertIsNone(services.verify_suggestion(token,'kheer'))
            self.assertIsNone(services.verify_suggestion(token+'x','milk'))
        with patch.object(services.time,'time',return_value=1001):self.assertIsNone(services.verify_suggestion(token,'milk'))

    def test_only_unknown_images_or_names_trigger_gemini(self):
        fallback={'status':'suggestions','source':'Google Gemini','message':'Confirm','candidates':[{'label':'Milk','requires_confirmation':True}]}
        with patch.object(app.services,'closest_food',return_value=fallback) as call,patch('zxingcpp.read_barcodes',return_value=[]):
            result=app.search_food(engine(),encoded_image=picture());self.assertEqual(call.call_count,1)
            self.assertEqual(result['image_analysis']['source'],'Google Gemini')
            call.reset_mock();app.search_food(engine(True),encoded_image=picture());call.assert_not_called()
            app.search_food(engine(),query='unfamiliar food');self.assertEqual(call.call_count,1)

    def test_barcodes_never_go_to_gemini(self):
        with patch.object(app,'barcode_lookup',return_value={'barcode':'12345678','found':False}) as barcode,patch.object(app.services,'closest_food') as gemini:
            app.search_food(engine(),query='12345678');gemini.assert_not_called();barcode.assert_called_once_with('12345678')
        with patch('zxingcpp.read_barcodes',return_value=[SimpleNamespace(text='12345678')]),patch.object(app,'barcode_lookup',return_value={'barcode':'12345678','found':False}),patch.object(app.services,'closest_food') as gemini:
            app.search_food(engine(),encoded_image=picture());gemini.assert_not_called()

    def test_unknown_external_raw_label_can_be_confirmed_without_dessert_match(self):
        with patch.object(app,'MODEL') as model:
            model.exists.return_value=False;lookup=app.Engine()
        lookup.model=SimpleNamespace(classes_=np.array(['doodhpak']))
        label='Fresh cow milk';token=services.suggestion_token(label,'raw ingredient')
        body=json.dumps({'label':label,'confirmation_token':token}).encode()
        handler=app.Handler.__new__(app.Handler);handler.path='/api/confirm-food';handler.headers={'Content-Length':str(len(body))};handler.rfile=io.BytesIO(body)
        captured=[];handler.send_json=lambda status,data:captured.append((status,data))
        with patch.object(app,'ENGINE',lookup),patch.object(app.services,'estimate_nutrition',return_value={'status':'unavailable','message':'Estimate not requested by this routing test.'}):handler.do_POST()
        self.assertEqual(captured[0][0],200)
        result=captured[0][1];self.assertEqual(result['recognition_source'],'Google Gemini')
        self.assertEqual(result['recipe_matches'],[]);self.assertEqual(result['name_matches'],[])
        self.assertEqual(result['ingredient_suggestion']['ingredients'][0]['name'],label)

    def test_secrets_and_auth_urls_do_not_leak_in_errors_or_barcode_results(self):
        url='https://api.barcodelookup.com/v3/products?key=dummy-private-secret'
        error=urllib.error.HTTPError(url,403,'dummy-private-secret',{},io.BytesIO(b'private'))
        with patch.object(services._HTTP,'open',side_effect=error):
            with self.assertRaises(services.ProviderError) as caught:services.api_json(None,'Barcode Lookup')
        self.assertNotIn('dummy-private-secret',str(caught.exception));self.assertNotIn('https://',str(caught.exception))
        payload={'products':[{'barcode_number':'12345678','title':'Product'}]}
        with patch.object(services,'configuration',return_value=CONFIG),patch.object(services,'api_json',return_value=payload):result=services.barcode_product('12345678')
        self.assertNotIn(CONFIG['barcode_key'],json.dumps(result))
        self.assertEqual(result['source_url'],'https://www.barcodelookup.com/12345678')

    def test_barcode_metadata_is_enriched_with_sourced_nutrition(self):
        product={'barcode':'12345678','found':True,'name':'Provider product','source_url':'https://www.barcodelookup.com/12345678'}
        nutrient={'barcode':'12345678','found':True,'name':'OFF product','source_url':'https://world.openfoodfacts.org/api/v2/product/12345678.json','nutrition':{'energy_kcal':123},'rating':{'grade':'B'}}
        with patch.object(services,'barcode_product',return_value=product),patch.object(app,'open_food_facts_lookup',return_value=nutrient):result=app.barcode_lookup('12345678')
        self.assertEqual(result['nutrition']['energy_kcal'],123);self.assertEqual(result['rating']['grade'],'B')
        self.assertIn('Open Food Facts',result['source']);self.assertIn('nutrition_source_url',result)
        with patch.object(services,'barcode_product',return_value=product),patch.object(app,'open_food_facts_lookup',return_value={'found':False}):result=app.barcode_lookup('12345678')
        self.assertIsNone(result['rating']['grade']);self.assertTrue(all(v is None for v in result['nutrition'].values()))

    def test_barcode_key_failure_keeps_open_food_facts_result(self):
        with patch.object(services,'barcode_product',side_effect=services.ProviderError('Barcode Lookup quota or rate limit reached.',429)),patch.object(app,'open_food_facts_lookup',return_value={'found':True,'name':'OFF product'}):result=app.barcode_lookup('12345678')
        self.assertTrue(result['found']);self.assertIn('quota',result['warnings'][0])

    def test_env_is_not_served(self):
        handler=app.Handler.__new__(app.Handler);handler.path='/.env';captured=[]
        handler.send_json=lambda status,data:captured.append((status,data));handler.do_GET()
        self.assertEqual(captured[0][0],404)

if __name__=='__main__':unittest.main()
