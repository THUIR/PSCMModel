# [PSCMModel](https://github.com/THUIR/PSCMModel)

PSCMModel is a small set of Python scripts for the user click models based on Yandex version (https://github.com/varepsilon/clickmodels). 

A *Click Model* is a probabilistic graphical model used to predict search engine click data from past observations.

This project is aimed to implement recently proposed click models and intended to be easy-to-read and easy-to-modify. If it's not, please let me know how to improve it :)

# Models Implemented
- *Partially Sequential Click Model* ( **PSCM** ) model: Chao Wang, Yiqun Liu, Meng Wang, Ke Zhou, Jian-Yun Nie, Shaoping Ma. Incorporating Non-sequential Behavior into Click Models. SIGIR (2015).
- *Temporal Hidden Click Model* ( **THCM** ) model: Danqing Xu, Yiqun Liu, Min Zhang, Shaoping Ma. Incorporating revisiting behaviors into click models. WSDM (2012).
- *Temporal Click Model* ( **TCM** ) model: Wanhong Xu, Eren Manavoglu, Erick Cantú-Paz. Temporal Click Model for Sponsored Search. SIFIR (2010).
- *Partially Observable Markov Model* ( **POM** ) model: Kuansan Wang, Nikolas Gloy, Xiaolong Li. Inferring search behaviors using partially observable markov (pom) model. WSDM (2010).
- *Dynamic Bayesian Network* ( **DBN** ) model: Chapelle, O. and Zhang, Y. 2009. A dynamic bayesian network click model for web search ranking. WWW (2009). (This model is exactly the same implementation as Yandex version)
- *User Browsing Model* ( **UBM** ): Dupret, G. and Piwowarski, B. 2008. A user browsing model to predict search engine click data from past observations. SIGIR (2008). (This model is exactly the same implementation as the inference method from the original paper, which is slightly different from Yandex version)

# Files
## README.md
This file.
 
## bin/
Directory with the scripts.

## sample/
Directory with the sample dataset.

## relevance_test_dataset/
query-result relevance data for PSCM model's paper (Section 5.3)

## user_preference_test_dataset/
user preference test data for PSCM model's paper (Section 5.4)

# Format of the Input Data 
A small example can be found under `sample/` (tab-separated). 5 files are included in this directory:

- query_id: encode each query into a unique id. 
  - e.g.: "test  1 10  5" means query ("test") with a unique id (1), 10 sessions are found in search logs and 5 sessions contain click action.
- query_class: The probability of been each searh intent for each query. 
  - e.g.: "test 0.25  0.25  0.25  0.25" means query ("test") has four search intents. Set "query_id 1" for each query if this information is needless.
- url_id: encode each URL into a unique id.
- train_data, test_data: search logs, in which each line represents one query-session. 10 tab-separated part are included. The inner separator for each [] is space (" "): 
  - query_id  [url_id * 10] [click * 10]  [click_time * 10] [mouse_feature_1 * 10] [mouse_feature_2 * 10]  [mouse_feature_3 * 10]  [mouse_feature_4 * 10]  [mouse_feature_5 * 10]  [mouse_feature_6 * 10]
  - click: 1 represents click, 0 represents no click
  - click_time: >0 represents click time in seconds, -1 represents no click
  - mouse_feature_1: The most left position mouse ever reach to in the result’s display area
  - mouse_feature_2: User’s total right towards movement length in the result’s display area
  - mouse_feature_3: The total dwell time that cursor spends in the result's display area neglecting its horizontal coordinate
  - mouse_feature_4: The total dwell time that cursor spends in the result's display area
  - mouse_feature_5: The total time that cursor hovers over the result's display area
  - mouse_feature_6: The amount of cursor actions (scroll, test select, move times) that appear in the result's display area
  - Ps: Just set mouse feature as 0 if your search logs do not contain mouse movement information.



# Usage
in bin/config_sample.py: select click models (e.g.: TEST_MODELS = ['PSCM', 'UBM', 'DBN', 'POM', 'TCM', 'THCM'])

in bin/ : python wc_click_model_inference_by_id.py ../sample

# Output
in target data directory, "/output" directory will be automatically generated, in which model results will be logged:
- model_name.model: Parameters of this model generated from train_data
- model_name.model.perplexity: Perplexity metrics of this model tested on test_data
- model_name.model.relevance: Query-URL-Relevance generated from this model 





