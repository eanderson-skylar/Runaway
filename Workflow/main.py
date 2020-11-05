import Model.FeatureExtraction

def train_model(model_name):
    model = FEModel(model_name=model_name)
    model.create_tokenizer()
    model.create_character_mapping()
    model.import_training_data()
    model.create_tv_arrays()
    model.train()

if __name__ == "__main__":
    train_model(model_name='Ad Runaway_all_feature v7')