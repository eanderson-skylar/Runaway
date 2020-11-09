from Model.FeatureExtraction import FEModel

fe = FEModel(model_name='Ad Runaway_all_feature v7')

def train_main_model():
    model = fe
    model.create_tokenizer()
    model.create_character_mapping()
    model.import_training_data()
    model.create_tv_arrays()
    model.train()

if __name__ == "__main__":
    train_main_model()