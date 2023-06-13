# MedNormalization
MedNormalization is a code for an Entity Linking model. It is my implementation, developed from scratch in 2022, based on this approach. The code includes all the necessary dataclasses and data preparation procedures.

# Usage
1. train.py. Example usage: ```python train.py -tr ./Demo/data/demo.json -val ./Demo/data/demo.json -ts ./Demo/data/demo.json -load_pretrained ./Model_weights/rubert_11072022_test -res ./Demo/saved_model_dir/ --use_cuda```
2. predict.py. Example usage: python predict.py -p ```./Demo/data/demo.json -res ./Demo/data/demo_with_predicted_meddra_ents.json -model ./Demo/saved_model_dir/```
3. norm_eval.py. Example usage: ```python norm_eval.py -t ./Demo/data/demo.json -p ./Demo/data/demo_with_predicted_meddra_ents.json```