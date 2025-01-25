@echo off
REM Find all .ipynb files matching 'HW*.ipynb' and process them

for /R %%f in (HW*.ipynb) do (
    jupyter nbconvert --clear-output --inplace "%%f"
    git add "%%f"
)