"""Small live Gemini/Barcode Lookup check. Prints outcomes, never credentials."""
import argparse
import time
from PIL import Image,ImageDraw
import external_services as services

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--barcode',action='store_true',help='Also use one Barcode Lookup request.')
    parser.add_argument('--nutrition-label',help='Check confirmation nutrition for this food name instead of image matching.')
    args=parser.parse_args()
    config=services.configuration()
    print({'gemini_configured':bool(config['gemini_key']),'barcode_lookup_configured':bool(config['barcode_key'])},flush=True)
    if args.nutrition_label:
        import io,json
        from unittest.mock import patch
        import fitness_food as app
        with patch.object(app,'MODEL') as model:
            model.exists.return_value=False;engine=app.Engine()
        label=args.nutrition_label
        body=json.dumps({'label':label,'confirmation_token':services.suggestion_token(label,'dish')}).encode()
        handler=app.Handler.__new__(app.Handler);handler.path='/api/confirm-food';handler.headers={'Content-Length':str(len(body))};handler.rfile=io.BytesIO(body)
        captured=[];handler.send_json=lambda status,data:captured.append((status,data))
        with patch.object(app,'ENGINE',engine):handler.do_POST()
        status,data=captured[0];profile=data.get('nutrition_profile')
        print({'http_status':status,'profile':profile,'warnings':data.get('warnings',[])},flush=True)
        return
    image=Image.new('RGB',(416,416),'#353535');draw=ImageDraw.Draw(image)
    draw.ellipse((25,70,395,380),fill='#909090');draw.ellipse((40,85,380,365),fill='#faf9f2')
    started=time.monotonic()
    try:
        model=services.gemini_model(config) if config['gemini_key'] else None
        print({'gemini_model':model,'metadata_seconds':round(time.monotonic()-started,1)},flush=True)
    except services.ProviderError as error:
        print({'stage':'Gemini model discovery','seconds':round(time.monotonic()-started,1),'message':str(error)},flush=True)
        return
    result=services.closest_food('White liquid in a cooking pan; identify the closest generic food.',image)
    print({'service':'Gemini','status':result['status'],'model':result.get('model'),
           'matches':[c['label'] for c in result['candidates']],'message':result['message'],'seconds':round(time.monotonic()-started,1)},flush=True)
    if args.barcode:
        try:
            product=services.barcode_product('3017620422003')
            print({'service':'Barcode Lookup','found':bool(product and product.get('found')),
                   'name':product.get('name') if product else None},flush=True)
        except services.ProviderError as error:print({'service':'Barcode Lookup','message':str(error)},flush=True)

if __name__=='__main__':main()
