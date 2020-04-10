cd data/json_bundle_news_domestic_violence
#gunzip -ck large-bundle-corona.json > reviews-180k-bundle-after-corona.json.gz
zip no_rep_news-domestic-no-violence.zip no_rep_news-domestic-no-violence.json 
zip no_rep_news-domestic-violence.zip no_rep_news-domestic-violence.json 

cd ..
cd neural_network_config
zip model_and_tokenizer.zip model.h5 model.json temp-model.h5 tokenizer.pickle 

