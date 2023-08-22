# On The Impact of Machine Learning Randomness on Group Fairness
The repository contains the code for the paper [On The Impact of Machine Learning Randomness on Group Fairness](https://dl.acm.org/doi/abs/10.1145/3593013.3594116) at FAccT 2023.

* [Downloading Datasets](#downloading-datasets)
* [Training a Model](#training-a-model)
* [Fairness Variance](#fairness-variance)
* [Creating Custom Data Orders](#creating-custom-data-orders)

## Downloading Datasets

#### Folktables Dataset

Install the folktables dataset package. (Check the official repo for more details - https://github.com/socialfoundations/folktables)
```
pip install folktables
```
No manual downloading is required beyond installing the folktables package. Datasets are downloaded automatically when running the code for the first time.

#### CelebA Dataset

Download the CelebA dataset. We recommend using the version present on Kaggle - https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.

For our experiments, we do not directly use the CelebA dataset, but instead only the 'features' extracted using a pre-trained ResNet50. A simple script to preprocess the CelebA dataset and extract all features is provided in `preprocess_celeba.py`. Please change the file locations according to dataset location before executing the script.

_Note: Make sure to also change the dataset location appropriately in `dataloader.py` (look for function `load_celeba_dataset`)._

The complete data downloading and preprocessing pipeline for CelebA can take upto a day.

## Training a Model

Train a model by executing `main_train.py`. Use the argparse help to understand the flags.
```
python main_train.py --help
```

_Note: The training setup saves model checkpoints at the end of every epoch. If you don't want that, change the boolean `save_every_ite` in `train_utils.py`._

**Example:** Execute the following to train a MLP with a single hidden layer (64 neurons) on ACSEmployment dataset while employing the reweighing technique to enforce fairness.
```
python main_train.py --ckptfldr ACSEmploymentDemo --arch mlp_64 --dataset acsemployment --num_feat 16 --reweigh
```

## Fairness Variance

The file `main_variance.py` contains minimal code to load and record fairness scores across different checkpoints. The code can be extended to perform more complicated analysis.

**Example:** Execute the following to measure fairness variance (Average Odds) of 50 different MLP models with changing random seeds on ACSIncome dataset.
```
python main_variance.py --ckptfldr ACSIncomeDemo --arch mlp_64 --dataset acsincome --num_feat 10
```

_Note: Before executing the code above, one obviously needs to train those 50 models._

## Creating Custom Data Orders

Play with custom data orders and see its impact on model fairness using the file `main_custom_order.py`. Details of setting the data order ratio are present in the file.

The custom data order is created using the following principles,
1. Batches are created using the exact ratio of each subgroup given by the user. These ratios can be used to create fair data orders, or create adversarial data orders.
2. As soon as the data points from the smallest minorities are finished (i.e., we cannot create perfect batches in the required ratio anymore), the rest of the data is shuffled and placed at the start of the data order.
3. Thus, the last batches of the data order contain data points according to the required ratio. This is powerful enough to significantly change the fairness scores of the model in only a single epoch of fine-tuning.

_Note: Current implementation of custom data order relies on loading the complete dataset on the disk at the same time. For bigger datasets (like image datasets), it is advisable to adapt the code and instead create custom data orders using only the index of the datapoints, instead of the actual input._
