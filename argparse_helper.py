import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="mlp_64", help="Model architecture ID (See models.py for choices)")
    parser.add_argument("--loss", default="ce", help="Loss Type for Training (ce | fairce)")
    parser.add_argument("--dataset", default="acsincome", help="Dataset (acsincome | acsemployment | celeba)")
    parser.add_argument("--num_feat", type=int, default=10, help="Number of Input Features in the Dataset (Needs to be supplied separately)")
    parser.add_argument("--pc", default="sex", help="Protected Class (sex | race); Only relevant for Folktables Datasets")
    parser.add_argument("--ckptfldr", default="ACSIncome2018CA", help="Folder for Saving Model Files. CHoose a Unique Name to Differentiate Between Multiple Experiments")

    parser.add_argument("--init_seed", type=int, default=0, help="Seed for weight initialization")
    parser.add_argument("--shuffle_seed", type=int, default=0, help="Seed for random reshuffling")

    parser.add_argument("--epochs", type=int, default=300, help="Number of Training Epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for Training")
    parser.add_argument("--lmbd", type=float, default=1, help="Combinatory Factor for Fair Loss; Only relevant for fairce loss")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout Percentage")

    parser.add_argument("--gpus", default="0,1", help="GPU Device ID")
    parser.add_argument("--cuda", action="store_true", help="Supply this flag to use the GPUs")

    parser.add_argument("--fairbatch", action="store_true", help="Use FairBatch during Training")
    parser.add_argument("--reweigh", action="store_true", help="Use Reweighing during Training")

    args = parser.parse_args()

    return args
