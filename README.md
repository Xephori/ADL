# Sign Language Word Recognition Group 6

## What This Project Does

We built a deep learning system that watches a short video of someone signing an American Sign Language (ASL) word and predicts which word is being signed. The model picks from a vocabulary of 10 words.

Sign language recognition can help bridge communication between the Deaf community and hearing people, making everyday interactions more accessible.

**Dataset**: WLASL (Word-Level American Sign Language) from Kaggle, top 10 classes, around 740 video clips total.

**Best result**: 38.89% top-1 test accuracy on 10 classes. Random chance would be 10%.


## How the Model Works

Each video is broken into 16 evenly spaced frames. A custom CNN looks at each frame individually and produces a 256-number summary of what the hand looks like. Those 16 summaries are then combined and passed to a classifier that predicts the sign word.

We designed and tested five different model architectures, all built from scratch:

| Model | Architecture | How it combines frames |
|---|---|---|
| Model A | Custom CNN + Average Pooling | Takes the average of all 16 frame summaries |
| Model B | Custom CNN + Bidirectional LSTM | Uses a recurrent network to read frames in order |
| Model C | Custom CNN + BiLSTM + Temporal Attention | Learns which frames matter most |
| Model D | ResNet18 (ImageNet) + BiLSTM + Attention | Uses a large pretrained image model |
| KeypointLSTM | MediaPipe Hand Landmarks + BiLSTM | Uses hand joint positions instead of pixels |

**Key finding**: Model A, the simplest one, gave the best test accuracy. With only 111 training videos, the more complex models overfitted to the specific people in the training set. Simple average pooling generalised better to new unseen signers.

**CNN backbone**: 4 residual blocks with skip connections, output channels of 32, 64, 128, and 256.


## Experiment Summary

We ran 10 iterations of experiments, each testing one new idea.

| Iteration | What we changed | Best model | Test accuracy |
|---|---|---|---|
| 1 | Started with 100 classes then reduced to 20 | Model A |  5% |
| 2 | Reduced to 10 classes (our baseline) | Model A | 11.76% |
| 3 | Used pretrained ResNet18 instead of custom CNN | Model D | 29.41% |
| 4 | Pretrained CNN on our own video frames | Model A |  17% |
| **5** | **Pretrained CNN on ASL Alphabet images** | **Model A** | **38.89%** |
| 6 | Froze the pretrained CNN during training | Model A | 11.11% |
| 7 | Used a much lower learning rate of 5e-5 | Model A | 0% (failed) |
| 8 | Used a mid learning rate of 1e-4 | Model A | 22.22% |
| 9 | Repeated Iteration 5 with 4 different random seeds | Model A | 17% to 39% |
| 10 | Used MediaPipe hand keypoints instead of video frames | KeypointLSTM | 33.33% |


## Final Model Performance

| Metric | Value |
|---|---|
| Top-1 Accuracy | 38.89% (7 out of 18 test videos correct) |
| Top-5 Accuracy | 61.11% (correct answer in top 5 predictions) |
| Macro F1-Score | 0.34 |
| Macro Precision | 0.34 |
| Macro Recall | 0.40 |
| Multi-seed mean accuracy | 27.8% plus or minus 9.2% across 4 seeds |
| Random chance baseline | 10% |

## Setup

### Step 1: Clone the repository

```
git clone https://github.com/Xephori/ADL.git
cd ADL
```

### Step 2: Install all required packages

```
pip install -r requirements.txt
```




### Step 3: Download the dataset

The dataset is not included in this repository because it is too large. There are two ways to get it.

---

**Option A: Download from Google Drive (recommended, no Kaggle account needed)**

The pre-processed dataset folder is already uploaded to our Google Drive alongside the model weights.

1. Open the Google Drive link: https://drive.google.com/drive/folders/197ggC4-ApmuxbLZdNhSSSCJg2R8iSJeF?usp=drive_link
2. Find the folder named `data` and download it
3. Place the downloaded `data` folder inside the root of this repository so the structure looks like this:

```
ADL/
  data/
    dataset/
      SL/
        accident/
        before/
        ...
    metadata.csv
    label_map.csv
```

You do not need to run `data/preprocessing.py`. The `metadata.csv` and `label_map.csv` files are already included.

---

**Option B: Download from Kaggle (requires a Kaggle account and API key)**

Step 1: Set up your Kaggle API key. Go to https://www.kaggle.com/settings, scroll to the API section, and click "Create New Token". This downloads a file called `kaggle.json`. Place it at `~/.kaggle/kaggle.json` on Mac/Linux or `C:\Users\<username>\.kaggle\kaggle.json` on Windows.

Step 2: Run the preprocessing script. It will automatically download the dataset and generate the metadata files.

```
python data/preprocessing.py
```

After this step you should have:

```
data/dataset/SL/        folder of .mp4 video files
data/metadata.csv       one row per video with its label and train/val/test split
data/label_map.csv      mapping from class name to numeric ID
```

### Step 4: Download the model weights

Download from Google Drive and place all the `.pth` files into the `saved_models/` folder.

**Google Drive link**: https://drive.google.com/drive/folders/197ggC4-ApmuxbLZdNhSSSCJg2R8iSJeF?usp=drive_link

The most important files are listed below.

| File | What it is |
|---|---|
| `iter5_asl_pretrain_model_a_best.pth` | Best model, Model A with ASL Alphabet pretraining, 38.89% test accuracy |
| `cnn_backbone_asl_pretrained.pth` | CNN backbone pretrained on 29,000 ASL Alphabet images |
| `iter5_asl_pretrain_model_b_best.pth` | Best Model B checkpoint |
| `iter5_asl_pretrain_model_c_best.pth` | Best Model C checkpoint |
| `iter3_resnet18_v1_model_d_best.pth` | Best Model D checkpoint (Iteration 3) |
| `iter10_mediapipe_v2_noise_keypoint_lstm_best.pth` | Best KeypointLSTM checkpoint |


## Training from Scratch

Follow these steps to fully reproduce the training from the beginning.

Step 1: Pre-extract video frames. This only needs to be done once. It reduces epoch time from 20 minutes down to under 60 seconds.

```
python -m src.preextract_frames
```

Step 2: Pretrain the CNN backbone on the ASL Alphabet dataset. This teaches the CNN to recognise hand shapes from a large and diverse set of 29,000 hand images before it sees any sign word videos.

```
python -m src.pretrain_cnn_asl --max-per-class 1000 --epochs 15
```

This saves the pretrained weights to `saved_models/cnn_backbone_asl_pretrained.pth`.

Step 3: Train Model A using the pretrained CNN backbone. This is the full video-level training run that reproduces our best result.

```
python -m src.train --model model_a --pretrained-cnn saved_models/cnn_backbone_asl_pretrained.pth --epochs 200 --lr 5e-4 --seed 42
```

The best checkpoint is saved automatically to `saved_models/` and the training log CSV is saved to `results/`.

Step 4: Evaluate the newly trained model on the test set. Replace the filename below with the actual checkpoint name produced in Step 3.

```
python -m src.evaluate --model model_a --weights saved_models/<your_checkpoint_name>_best.pth --split test
```

## Reproducing Other Iterations

**Iteration 3 (Model D with ResNet18)**

```
python -m src.train --model model_d --epochs 100 --lr 1e-3 --seed 42
```

**Iteration 9 (Multi-seed robustness check)**

Run the Iteration 5 command three times, changing only the seed value each time.

```
python -m src.train --model model_a --pretrained-cnn saved_models/cnn_backbone_asl_pretrained.pth --epochs 200 --lr 5e-4 --seed 1

python -m src.train --model model_a --pretrained-cnn saved_models/cnn_backbone_asl_pretrained.pth --epochs 200 --lr 5e-4 --seed 7

python -m src.train --model model_a --pretrained-cnn saved_models/cnn_backbone_asl_pretrained.pth --epochs 200 --lr 5e-4 --seed 123
```

**Iteration 10 (MediaPipe hand keypoints)**

First extract the hand keypoints from all videos:

```
python -m src.extract_keypoints
```

Then train the keypoint LSTM model:

```
python -m src.train_keypoints --epochs 100 --seed 42
```

## Key Training Arguments

| Argument | Default | What it controls |
|---|---|---|
| `--model` | model_a | Which architecture to train. Options are model_a, model_b, model_c, model_d |
| `--lr` | 5e-4 | Learning rate |
| `--epochs` | 200 | Maximum number of training epochs |
| `--batch-size` | 8 | Number of videos per batch |
| `--num-frames` | 16 | Number of frames sampled per video |
| `--pretrained-cnn` | None | Path to a pretrained CNN backbone file |
| `--freeze-cnn` | False | If set, freezes the CNN and only trains the temporal layers |
| `--seed` | 42 | Random seed for reproducibility |
| `--run-name` | model name | Prefix used for the saved log file and checkpoint file |

Training logs are saved to `results/<run-name>_log.csv`.
Best checkpoint is saved to `saved_models/<run-name>_best.pth`.



## Running the Live Demo

The demo is a Gradio web app that lets you pick a test video and see the model predict the sign word in real time.

Step 1: Make sure `iter5_asl_pretrain_model_a_best.pth` is inside `saved_models/` and the frames have been pre-extracted by running `python -m src.preextract_frames`.

Step 2: Run the demo.

```
python app.py
```

This opens a browser window. Select a test video from the dropdown menu, click Run Model, and you will see the 16 sampled frames displayed as an animated GIF along with the top 5 predicted sign words and their confidence scores.



## Default Hyperparameters

All hyperparameters are stored in `configs/default.yaml` and can be overridden using command-line arguments when running `src/train.py`.

| Parameter | Value |
|---|---|
| Optimiser | Adam with beta1 0.9 and beta2 0.999 |
| Learning rate | 5e-4 |
| Batch size | 8 |
| Epochs | 200 |
| Loss function | Cross-Entropy with label smoothing 0.1 |
| Dropout | 0.3 |
| Gradient clipping | max norm 1.0 |
| Frames per video | 16 |
| Image size | 224 by 224 pixels |
| Early stopping patience | 100 epochs |
| Learning rate scheduler | StepLR, halves every 50 epochs |
| Data augmentation | Random horizontal flip during training |


## Package Dependencies

Python 3.10 or higher is required.

| Package | Version |
|---|---|
| torch | 2.1.0 |
| torchvision | 0.16.0 |
| numpy | 1.26.2 |
| pandas | 2.1.3 |
| scikit-learn | 1.3.2 |
| matplotlib | 3.8.2 |
| seaborn | 0.13.0 |
| opencv-python | 4.8.1.78 |
| Pillow | 10.1.0 |
| PyYAML | 6.0.1 |
| tqdm | 4.66.1 |
| kagglehub | 0.3.12 |
| jupyter | 1.0.0 |
| gradio | 4.0.0 or above |
| mediapipe | 0.10.0 or above |

Full list with pinned versions is in `requirements.txt`.



## Links

GitHub repository: https://github.com/Xephori/ADL.git

Google Drive (model weights and dataset): https://drive.google.com/drive/folders/197ggC4-ApmuxbLZdNhSSSCJg2R8iSJeF?usp=drive_link

Dataset source: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
