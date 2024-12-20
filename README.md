# Pushing Language Models Beyond Semantics: Isolating Authorship from Content with Contrastive Learning
This is the repository for the article Pushing Language Models Beyond Semantics: Isolating Authorship from Content with Contrastive Learning, available at arXiv [https://arxiv.org/abs/2411.18472](arxiv.org/abs/2411.18472)

**Purpose**: Authorship has entangled style and content inside. Authors frequently write about the same topics in the same style, so when different authors write about the exact same topic the easiest way out to distinguish them is by understanding the nuances of their style. Modern neural models for authorship can pick up these features using contrastive learning, however, some amount of content leakage is always present. Our aim is to reduce the inevitable impact and correlation between content and authorship. 

**Methods**: We present a technique to use contrastive learning (InfoNCE) with additional hard negatives synthetically created using a semantic similarity model. This disentanglement technique aims to distance the content embedding space from the style embedding space, leading to embeddings more informed by style. 

**Results**: We demonstrate the performance with ablations on two different datasets and compare them on out-of-domain challenges. Improvements are clearly shown on challenging evaluations on prolific authors with up to a 10% increase in accuracy when the settings are particularly hard. Trials on challenges also demonstrate the preservation of zero-shot capabilities of this method as fine tuning.


# Training the model yourself
This script trains a model that detangles author style and semantics using PyTorch Lightning. The model is based on the `SyntheticHardNegativesLossNonZeroSelfModel` and utilizes the WandbLogger for logging.

## Requirements

- Python 3.x
- PyTorch Lightning
- Wandb
- Transformers
- Other dependencies as specified in the script

## Usage

To run the script, you may use the following command:
```bash
python train_script.py --batch_size 128 --minibatch_size 8 --lr 0.005 --num_epochs 1 --warmup_steps 0.1 --max_length 512 --dropout 0.1 --name debug-run --unfrozen_layers 24 --reset_layers 0 --eps 1 --gamma 1 --with_weights --gpu 0 --dataset blog --log_steps max
```
