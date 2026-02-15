# [ISBI'26] From Specialist to Generalist: Unlocking SAM's Learning Potential on Unlabeled Medical Images

<img width="1338" height="698" alt="image" src="https://github.com/user-attachments/assets/4bbb78f9-9472-48da-bea3-25a2feb1c762" />


## Setup

Use **Python 3.10** and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the **PROMISE12** and **COLON** datasets and update the dataset paths inside `run.py`.

---

## Train

```bash
python run.py
```

---

## Test

Update the pretrained paths of SAM and Unet `run.py` then run:

```bash
python run.py --mode test
```
