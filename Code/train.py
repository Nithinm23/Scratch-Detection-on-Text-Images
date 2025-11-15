import yaml, os, random
from model import build_model
from dataset import MultiTaskSequence
from augmentations import get_train_aug, get_val_aug
from utils import split_list

def train():

    with open("./configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    root = cfg["paths"]["data_root"]
    good = sorted(os.listdir(f"{root}/good"))
    bad  = sorted(os.listdir(f"{root}/bad"))

    good_paths = [f"{root}/good/{f}" for f in good]
    bad_paths  = [f"{root}/bad/{f}" for f in bad]

    good_train, good_val, good_test = split_list(good_paths)
    bad_train,  bad_val,  bad_test  = split_list(bad_paths)

    train_samples = [(p,0) for p in good_train] + [(p,1) for p in bad_train]
    val_samples   = [(p,0) for p in good_val]   + [(p,1) for p in bad_val]

    random.shuffle(train_samples)

    train_gen = MultiTaskSequence(train_samples, f"{root}/masks",
                                  batch_size=cfg["training"]["batch_size"],
                                  tfm=get_train_aug())

    val_gen   = MultiTaskSequence(val_samples, f"{root}/masks",
                                  batch_size=cfg["training"]["batch_size"],
                                  tfm=get_val_aug())

    model = build_model()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg["training"]["epochs"]
    )

    model.save("./saved_models/final_model.keras")
    print("Model saved!")

if __name__ == "__main__":
    train()
