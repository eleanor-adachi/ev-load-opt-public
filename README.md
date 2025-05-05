# ev-load-opt-public
Public repository for EV load optimization research at UC Berkeley

## Project Description
This project aims to quantify the trade-offs between distribution grid upgrades and dispatchable generation and storage capacity procurement in the optimization of future electric vehicle (EV) charging demand in the service territory of Pacific Gas and Electric Company (PG&E) in Northern California. This project builds upon previous work by [Elmallah et al. (2022)](https://iopscience.iop.org/article/10.1088/2634-4505/ac949c/meta). This project uses convex optimization methods and assesses several different scenarios of load growth in 2030, 2040, and 2050 to explore both near- and long-term impacts. This project also estimates the costs of both distribution grid upgrades and capacity procurement under different objective functions.

## Acknowledgements
This project was primarily conducted by Eleanor Adachi between 2023 and 2025 to satisfy the requirements for the Master of Science degree in the Energy and Resources Group (ERG) at the University of California, Berkeley. This project was advised by Duncan Callaway and Maryam Mozafari and was built upon previous work by ERG researchers including Anna Brockway, Salma Elmallah, and Victor Reyes. Valuable feedback and assistance was provided by the Energy Modeling, Analysis, and Control (EMAC) group, particularly Ana Santasheva, Eli Brock, and Sunash Sharma. Tyler Nam, Paula Gruendling and Achintya Madduri were consulted as subject-matter experts.

## Getting Started

1. Clone the repo
   ```sh
   git clone https://github.com/eleanor-adachi/ev-load-opt-public.git
   ```
2. Create conda environment
   ```sh
   conda env create -f environment.yml
   ```
3. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```
4. Create a user account to access the PG&E Integration Capacity Analysis (ICA) map (you will be prompted for your username and password when running `download_pge_feeder_timeseries.py`): https://www.pge.com/b2b/distribution-resource-planning/integration-capacity-map.shtml
5. Activate conda environment
   ```sh
   conda activate ev_load_opt_env
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How to Use

The scripts and Jupyter Notebooks in this project relate to each other according to the process flow diagram below. Raw data is either automatically downloaded when you clone the repo or will need to be downloaded per the "*_download_instructions.txt" (where * is replaced with the name of a specific dataset) files saved in relevant raw data folders. (Some datasets could not be stored in this repo due to size limitations.)
![Redesigned V1G process map_for export](https://github.com/user-attachments/assets/9a8147a4-a5b3-4c15-a54b-6d49031300ef)

