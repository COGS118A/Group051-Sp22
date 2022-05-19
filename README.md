# COGS118A Project template
This is your group repo for your final project for COGS118A

This repository is publicly visible! We will be using that feature to do peer review of projects.

Template notebooks for each component are provided. Only work on the notebook prior to its due date. After each submission is due, move onto the next notebook (For example, after the proposal is due, start working in the Data Checkpoint notebook).

This is your repo. You are free to manage the repo as you see fit, edit this README, add data files, add scripts, etc. So long as there are the four files above on due dates with the required information, the rest is up to you all.

Also, you are free and encouraged to share this project after the course and to add it to your portfolio. Just be sure to fork it to your GitHub at the end of the quarter!

# Note to Graders
> **All submitted reports (i.e. `Checkpoint_group051.ipynb`) can be found in `/reports/`!**

# Group Notes
This project uses a trimmed down version of cookiecutter data science! It's designed to have sensible defaults as to where to place files. The biggest take-aways are the following:
- **Data is immutable**. Don't edit the downloaded data files directly. Rather, make copies and edit those copies.
- **Notebooks should be used for exploration/communication**. Code that we can all use is better to go in separate python files (located in `/src`).
  - Side note: use `%load_ext autoreload` and `%autoreload 2` in notebooks to automatically pull changes in edited python files.

## Setup
1. Install the dependencies. This can be done in one of two ways:
   - Using [poetry](https://python-poetry.org/)
   - Using the default Conda interpreter
2. Download the Kaggle dataset
   - This requires a Kaggle account!
   - Login to your Kaggle account, then click the **Download** button [here](https://www.kaggle.com/datasets/koryakinp/chess-positions?resource=download).
3. Unzip the downloaded `archive.zip` into `/data/raw`.
   - This should result in two separate folders: `/data/raw/train` and `data/raw/test`
4. Run the conversion script!
   - Out of concern for our hard drives, I've limited the script to only convert 10k test & 2k train
     - Converting everything takes up >>100GB!
     - Using this much data should give our models more then enough data while staying >15GB
   - If using poetry, `poetry run python src/data/make_dataset.py data/raw/ data/processed`
   - If using Conda, `python src/data/make_dataset.py data/raw/ data/processed`