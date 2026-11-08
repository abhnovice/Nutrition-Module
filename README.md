# Fitness Journal: local food search prototype

One Python service and one HTML test section. Food recognition runs locally using a frozen MobileNetV2 encoder and a food classifier trained on Indian dishes plus the raw-food dataset described below. Unrecognized foods use the Google Gemini API when configured. Barcode identification uses Barcode Lookup, with Open Food Facts nutrition enrichment. API keys remain on the Python server; no database or frontend build is needed.

## Run

Dependencies and the encoder have already been installed/downloaded in this workspace. Start or restart the server:

```bash
source .venv/bin/activate
python fitness_food.py serve
```

Open http://127.0.0.1:8000. Enter a food name or barcode, select an image, or combine a name with an image. Barcode photos use barcode decoding before food classification. A photo is either rejected as **Unrecognized or uncertain food**, or offers candidate labels. **Confirm a food label** before viewing its nutrition and recipe ingredients. Photo candidates alone carry no nutrition or ingredients. You can always use **Your own ingredients** or type a corrected food name; the workbook may lack raw-ingredient nutrition.

## Install / retrain on another machine

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
curl --fail --location --output mobilenetv2-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx
python prepare_raw_food.py
python fitness_food.py train
```

`python fitness_food.py refine` runs the same current training procedure. When the raw-food manifest exists, both commands use `train_combined.py`. `python fitness_food.py evaluate` reports separate dish/raw test metrics, including uncertainty coverage, for the combined model. `training_report.json` includes both raw accuracy and rejection coverage/conditional accuracy.

The 13.3 MB encoder is downloaded once from the [official ONNX model repository](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet), under Apache 2.0. Its SHA-256 is pinned in the Python script and checked before loading. It uses ImageNet normalization and a 1,280-dimensional pooled representation; only the local food SVM is fitted, rather than fine-tuning encoder weights. ONNX Runtime uses CPU execution. The trained classifier runs locally. Unrecognized images can be sent to Gemini as metadata-free JPEG thumbnails up to 1024 px; recognized images stay local. The pretrained weights download does not upload images. See [ONNX Runtime documentation](https://onnxruntime.ai/docs/api/python/api_summary.html).

## Gemini fallback and barcode API configuration

The server reads `.env` on each request. Process environment values take precedence. `.env` is ignored by Git, is not a static route, and is never included in browser responses. `.env.example` shows the supported variable names:

```dotenv
GEMINI_API_KEY=your_google_gemini_key
# Optional override; otherwise choose an available Flash model from the account.
GEMINI_MODEL=
# Key issued by barcodelookup.com
BARCODE_API=your_barcode_lookup_key
```

`GOOGLE_API_KEY` is also accepted for Gemini, and `BARCODE_API_KEY` for Barcode Lookup. Restart the Python service after a code update; later saved key changes are read without restarting. Existing model weights do not need retraining for this API feature.

Search first checks typed/decoded barcodes and local food matches. **Uncertain image recognition or an unmatched food name triggers Gemini once per search.** Only unrecognized images are uploaded, resized to at most 1024 px with metadata removed. Recognized local images do not call Gemini. Food context typed in the search is sent with an unknown image; unknown name-only searches send text. This uses Gemini multimodal inference, not a Google Search web-grounding tool.

Gemini returns up to three closest food labels and short explanations. Nonfood/ambiguous results remain unrecognized. The UI marks cloud suggestions and requires confirmation before local nutrition/recipe lookup. Suggestions can name foods outside the trained classes. Confirmation uses a server-signed token that expires after 15 minutes; restart or expired-token results require a fresh search. Raw ingredient suggestions use exact nutrition matching, with manual input for missing values, and do not turn into dessert recipes. The recognition call returns food names only. On confirmation, the server first resolves an exact local food name or translation alias. If a suitable profile is missing, it asks Gemini for a separate generic nutrition estimate. Estimates are visibly labeled, include preparation/assumptions, and show approximate values per 100 g of food or 100 ml of beverage. Plain tea/coffee default to brewed, unsweetened drinks made with water. Brand-specific package data is not inferred. The A–F prototype rating is calculated locally from available nutrients and labeled as based on estimated values; an official Nutri-Score is never generated by Gemini. Provider failures preserve available local data and display a retryable explanation. Local model validation accuracy does not measure Gemini quality.

[Gemini API documentation](https://ai.google.dev/api/generate-content) describes the REST image and JSON response format; [model discovery](https://ai.google.dev/api/models) selects an available Flash model unless `GEMINI_MODEL` is set. Authentication uses `x-goog-api-key` on the fixed Google API endpoint. Generation has a 45-second timeout; model metadata is cached in memory for one hour. Automatic discovery prefers an available lightweight Flash model. A generation HTTP 502/503/504 can trigger one alternative model attempt; explicit GEMINI_MODEL settings, quota errors and authentication errors never trigger model switching. Quota, authorization, network and malformed-output failures become explicit messages while local results stay available. Keys and provider error bodies are never printed or returned.

For a typed or decoded barcode, [Barcode Lookup](https://www.barcodelookup.com/api-documentation) identifies the product using `BARCODE_API`. Its endpoint authenticates with a key query parameter, so that private request URL is never returned or logged. Public product links contain only the barcode. Requests carrying API credentials reject redirects. Open Food Facts supplies separately attributed nutrients and official Nutri-Score when present. A product identity without nutrition remains unrated, with missing nutrient fields. Barcode Lookup outages fall back to Open Food Facts. If no barcode key is configured, the original Open Food Facts route remains available.

For a small live check (uses API requests):

```bash
python check_external_services.py
# Optionally make one Barcode Lookup request for a public Nutella barcode too:
python check_external_services.py --barcode
```

The check uses a synthetic white-liquid image and prints only outcome metadata, never credentials. Unit tests mock provider calls and do not spend API quota. Live checks passed for Barcode Lookup (public Nutella product) and Gemini image inference (gemini-3.1-flash-lite, returning milk/cream suggestions). Gemini 3.8 Flash returned HTTP 503 during the earlier check; automatic selection uses the available lighter model. These are connectivity/integration checks, not a recognition accuracy benchmark. Browser visual checks were not performed for this change.

## Model refinement and uncertainty

The **earlier dish-only model lacked milk** in its 81 classes; doodhpak, kheer and other dairy dishes were included. That forced-choice classifier could not reliably label a milk photo. The combined model adds a distinct `milk` class and other raw-food classes from the new labeled pictures. Adding a recipe CSV does not add milk image training examples. This is a class-coverage problem as well as a feature/generalization problem. The earlier handcrafted-feature model achieved approximately 99.7% training accuracy versus 30% validation accuracy.

The earlier dish-only run used 3,087 full training photos, grouped by filename using `_annotations.csv`; the combined run adds the audited raw-food training images. For photos with multiple labels, the label with the largest summed annotation area is used for training. Full images are encoded consistently during training and inference. The earlier run selected C=3 from 0.3, 1, 3 and 10; the combined pipeline compares C=1, 3 and 10. No test images enter classifier fitting, correctness calibration or threshold selection.

A separate logistic correctness estimator learns from training-only three-fold out-of-fold predictions. It estimates whether the top label is correct; it does **not** turn class rank scores into per-food probabilities. The UI displays ranked options rather than misleading vote percentages. Validation selects a reliability cutoff with a minimum 50% correctness estimate, at least 15 accepted examples and at least 70% observed accuracy among accepted examples. Recognition also requires agreement between whole-photo and center-crop labels and sufficient similarity to training examples (nearest-neighbor distance cutoff derived from training). When no cutoff qualifies, automatic suggestions are disabled. Older artifacts without this policy require `refine` before suggestions are enabled.

The reject policy is evaluated on known dish and raw-food classes, not on a dedicated set of unfamiliar foods. It is a heuristic, not proof that an image belongs to a known class. An unfamiliar food can still resemble a known dish. Confirmations therefore remain mandatory. Identifying plain milk reliably requires appropriately labeled milk photos and diverse negative examples, including milk desserts and different pans/containers. Milk classification is trained by the combined pipeline; ingredient segmentation is still not trained. Its performance on milk in kitchen pans must be tested separately from the dataset split.

`model.npz` stores numeric feature arrays, labels and reproducible model/policy settings (no executable pickle). The server reconstructs the classifier and nearest-neighbor index on startup. The pre-refinement model and report are retained as `model_before_refinement.npz` and `training_report_before_refinement.json`.

## Raw-food training integration

Source: [ibrahimdaud/raw-food-recognition](https://huggingface.co/datasets/ibrahimdaud/raw-food-recognition), pinned to revision `e03922c47df5c8015616e3c88536883a1aacd10a`. `prepare_raw_food.py` uses `datasets.load_dataset` with this revision, keeps the Hugging Face cache inside `.hf_cache`, and writes normalized local images plus a provenance manifest under `Datasets/raw-food-recognition/`. No API credentials are required for this public dataset. Downloads are only needed for dataset preparation; training and the first-pass classifier run locally afterward. Gemini fallback and barcode product lookups need internet access.

These are image-level class labels, **not ingredient bounding boxes or segmentation masks**. Class names replace underscores with spaces and standardize `yoghurt` to `yogurt`; dairy milk, soy milk, oat milk and flavored yogurts remain distinct.

The source training split remains training data. Exact-content groups in the source validation split are deterministically divided per class into separate tuning and evaluation sets. The original Indian dish train/valid/test separation remains in place. Exact duplicates and conflicting image labels are removed; same-class near-duplicates across splits are conservatively excluded using a 64-bit perceptual hash within Hamming distance four. Test images take precedence over validation, which takes precedence over training when excluding duplicates. Classes with fewer than three remaining training examples are excluded and reported. This heuristic can miss near duplicates or discard similar but distinct images.

`Datasets/combined_split_manifest.json` records the final assignment, exclusions, content hashes and source revision. `Datasets/combined_features.npz` caches image embeddings; the cache fingerprint includes encoder hash, split records and local file size/modification time. The classifier, correctness calibration and novelty distance model use training data only. Joint-classifier validation trials may be reused when the image-feature fingerprint, encoder, model parameters, library versions and audited split metadata are unchanged. The trainer also compares a raw/dish router with separate classification heads against the joint classifier, selecting on validation only. Hyperparameters maximize the mean of raw-food and dish macro class validation accuracy so the larger raw dataset cannot hide a regression on dishes. The shared rejection cutoff must meet the minimum sample count and accepted-accuracy target separately in both domains. Test is evaluated after freezing those choices. Raw test is carved out of source validation; both dish and raw evaluation splits have now been inspected across development rounds and are not fresh external blind benchmarks.

The combined artifact stores the list of raw classes. After confirmation of a raw class, the service looks for an **exact** raw-ingredient nutrition record and offers a single editable ingredient row. It does not substitute soup, dessert or other fuzzy prepared-dish matches and does not infer a recipe from a raw label. The image dataset does not provide nutrition; absent nutrients remain unknown and require a suitable reference. Milk type, amount, cooking changes and mixtures still require confirmation.

Training preserves the earlier model/report as `model_before_raw_food.npz` and `training_report_before_raw_food.json`. The complete combined evaluation and per-class confusion counts, including milk/doodhpak/kheer, are saved in `training_report.json`. Source metadata tags the repository MIT, while its card describes educational use and refers to the original source datasets for their terms.

## Combined model results

The selected model is the joint RBF SVM with C=3, trained on **11,176 images** across **169 classes** (81 dishes and 88 normalized raw-food classes). Duplicate filtering removed 876 exact raw duplicate copies, 1,109 same-class cross-split near duplicates, and 12 conflicting-label source images. All 12,505 source images were readable. Milk has 298 training images, 37 validation images and 37 test images.

| Cleaned evaluation set | Images | Top-1, all images | Top-3, all images | Retained coverage | Accuracy among retained images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw food | 1,215 | 91.85% | 97.20% | 90.04% | 96.34% |
| Dishes | 379 | 55.15% | 77.57% | 59.89% | 70.93% |

Milk was correctly classified on **37/37** test images and retained on all 37. This is a small, source-specific classification result. The milk examples inspected during preparation depict grocery-store cartons; it does not demonstrate recognition of poured milk in a pan. Image-level classification does not establish ingredient segmentation.

Earlier dish scores used a less strict duplicate audit. On the cleaned validation data, an independently trained dish-only head scored 52.54%, compared with 51.98% for the joint model. A learned raw/dish router with separate heads did not beat the joint classifier on the mean of domain-specific macro validation accuracy, so it was not selected. The full trials and per-class confusion counts are in `training_report.json`. Earlier benchmark numbers below are historical and should not be read as the current cleaned-data performance.

## Earlier dish-only refinement results

| Metric | Earlier handcrafted model | MobileNet feature model |
| --- | ---: | ---: |
| Validation top-1, all 360 photos | 30.00% | 65.83% |
| Validation top-3, all photos | 37.78% | 78.89% |
| Test top-1, all 379 photos | 29.29% | 64.12% |
| Test top-3, all photos | 40.37% | 83.91% |

The rejection policy retained 65.0% of validation photos with 79.91% conditional top-1 accuracy, and 63.59% of test photos with 80.50% conditional top-1 accuracy. Rejected photos receive no image-based dish profile. Those earlier statistics are preserved in `training_report_before_raw_food.json`; conditional accuracy must always be read together with coverage. The provided test set has been reused across development rounds and is not a fresh blind benchmark. Earlier split auditing found no shared source filenames or identical decoded images; near duplicates and other dataset biases remain possible. These are dish-classification measurements, not ingredient-segmentation or unseen-milk recognition measurements.

## Checks

```bash
python -m unittest -v
python -m py_compile fitness_food.py test_fitness_food.py
```

All 44 unit tests passed, including nutrition confirmation and external-service routing tests. The unit suite covers raw-label normalization, deterministic split assignment, duplicate isolation, conservative raw nutrition matching, and confirmation routing, alongside the earlier checks. They cover existing nutrition/barcode behavior, ingredient profiles, uncertainty rejection, crop disagreement, unfamiliar-image distance, and mandatory confirmation. Python and frontend JavaScript syntax checks pass. Synthetic blank, noise and pan-with-white-liquid images were rejected in smoke checks; they do not establish performance on the user's original milk photo, which was not available for testing. The saved combined classifier passed a real milk-image search and raw confirmation/name-search routing check. Its synthetic white-liquid pan was rejected without an automatic dish profile. Browser visual checks were not performed.

The server binds only to localhost. The page accepts images up to 8 MB; the server accepts JSON requests up to 12 MB. Images are not persisted.

## Nutrition and ratings

Workbook: `Datasets/Anuvaad_INDB_2024.11.xlsx`, 1,014 food records. Base nutrient columns are treated as per 100 g; the provided workbook does not embed a units reference, so verify this basis against the original dataset documentation before relying on values. Serving columns are not used. Saturated fat is converted from mg to g; missing values stay missing. Nutrition describes the matched recipe, not the photographed meal.

There is no standard A–F Nutri-Score. Official Nutri-Score uses **A–E**. Barcode results preserve the grade supplied by Open Food Facts, or show Unrated. Local workbook records lack total sugars, fruit/vegetable composition and category information needed for a faithful standard calculation. They therefore use an explicitly experimental A–F rating, not an official grade:

`score = kcal/100 + saturated_fat_g/2 + sodium_mg/200 + free_sugar_g/5 - min(fibre_g/2,5) - min(protein_g/5,3)`

Thresholds: A ≤ 0; B ≤ 2; C ≤ 4; D ≤ 6; E ≤ 9; F > 9. All inputs refer to 100 g. These are custom prototype choices, not validated nutrition guidance. Missing required inputs yield no rating. Free sugar and total sugar are deliberately kept distinct.

Barcode photos are decoded locally by ZXing. Decoded numeric codes use Barcode Lookup when its key is configured, followed by `https://world.openfoodfacts.org/api/v2/product/{barcode}.json` for nutrition. Decoded barcodes bypass Gemini. Images are processed in memory and not saved; unknown food photos may be uploaded to Gemini as described below. Open Food Facts data is community maintained and licensed under ODbL; source links appear in results. A network outage is reported without discarding local name results.

References: [official Nutri-Score](https://www.santepubliquefrance.fr/nutrition-et-activite-physique/nutri-score), [Open Food Facts API](https://openfoodfacts.github.io/openfoodfacts-server/api/). Review dataset source licenses before distributing trained artifacts or nutrition data.

## Ingredient breakdown and custom profiles

`Cleaned_Indian_Food_Dataset.csv` adds 5,938 searchable recipe ingredient lists. Name searches and image dish candidates now include up to three suggested recipes. Expand an ingredient suggestion, confirm the recipe, edit ingredient names, exclude unused rows, and enter measured weights in grams. Use **Add ingredient** for additional rows, or expand **Your own ingredients** to build a profile manually. Original recipe quantity text is shown as a reference; cups, spoons, counts, cooking loss, and hidden quantities are not automatically converted to grams.

The photo classifier identifies a dominant dish or a trained raw-food class. Ingredient lists come from recipe title retrieval using local word TF-IDF and required title words, not ingredient object detection or segmentation. The recipe CSV includes remote image URLs but no local ingredient-labeled pixels, bounding boxes, or ingredient nutrition. The added Hugging Face data provides image-level raw-food labels, also without masks or boxes. No images are fetched from these URLs. True automatic ingredient localization would require an ingredient detector/segmenter trained with appropriate ingredient-level supervision. Current dish recognition accuracy also limits recipe suggestions. Confirm or correct the dish name before using a recipe.

Raw ingredient nutrition first uses an exact matching local food name or translation alias. Confirmation also populates a separate nutrition card, using a labeled Gemini estimate when needed. Fuzzy prepared-dish matches are deliberately not substituted for raw ingredients. For per-100-g estimates, the ingredient editor is filled automatically. Beverage estimates per 100 ml stay on the nutrition card; copying them into the gram-based calculator requires a density conversion. Nutrients the provider cannot estimate remain missing and can be entered from a suitable reference or package. Enter nutrient values **per 100 g**, including free sugar rather than total sugar. Editing the ingredient identity requires checking its nutrient values too.

`POST /api/ingredients/profile` accepts an `ingredients` list with `name`, `grams`, and `nutrition_per_100g`. It returns individual weighted contributions, whole-mixture totals, values normalized per 100 g of input weight, missing-input details, and the existing prototype rating. Whole-mixture nutrient totals are unavailable if any included ingredient is missing that nutrient; missing values are never counted as zero. A rating appears only when all required rating inputs are supplied. Input weight is not final cooked yield, and results are estimates from user-confirmed inputs rather than measurements of a photograph.

Ingredient update verification: unit tests cover ingredient parsing, conservative nutrition matching, weighted totals, missing values, invalid inputs, and both API handlers using in-memory requests. Python compilation and frontend JavaScript syntax checks passed. Browser visual checks and real ingredient-segmentation accuracy checks were not performed; there is no ingredient segmentation model in this version. The recipe-retrieval feature itself does not train ingredient vision; the later model refinement retrains dish recognition only.

## Confirmation nutrition verification

Clicking a confirmed local or Gemini suggestion returns `nutrition_profile` from `/api/confirm-food` and immediately renders its nutrition card. Fuzzy catalog matches are not substituted for the confirmed food: green tea cannot silently use sweetened milk tea values. The page displays a loading message while resolving the profile, source attribution, preparation assumptions, and an approximate-value marker for model estimates.

A live check of `python check_external_services.py --nutrition-label "green tea"` returned HTTP 200 with a populated Gemini estimate for brewed, unsweetened green tea per 100 ml, including energy, protein, carbohydrate, fat, sugars, fibre and sodium. This verifies integration, not the accuracy of brand-specific nutrition. Unit tests check local-record preference, alias matching, beverage units, zero values, invalid estimates, raw-editor population and provider outages. Python compilation and JavaScript syntax checks passed; browser visual checks were not performed.
# Nutrition-Module
