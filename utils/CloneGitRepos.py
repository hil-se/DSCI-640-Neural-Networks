
# Automation code for cloning all the student repositories once you have access to them

# Pre-requisites: Packages to be installed prior to running this code for cloning the repositories are mentioned below:
# 1. install GitPython

# Imp Note: Make sure you have access to all the repositories first

import git
import pandas as pd

if __name__ == "__main__":
    # Reads the excel file from the file path specified
    data = pd.read_csv("repos.csv")
    # Retains only RIT username and Github repo link columns from the excel data
    df = pd.DataFrame(data, columns=['RIT username','Github Repo'])
    # Iterate through each row in the df
    for row in df.iterrows():
        # Getting RIT username
        rit_username = row[1]['RIT username']
        # Getting the student github repo link
        git_hup_repo = row[1]['Github Repo']
        # Cloning the student github repo in the current directory and renaming it with rit username
        try:
            git.Repo.clone_from(git_hup_repo, rit_username)
        except:
            print("Repo not exist: %s" %git_hup_repo)

    print("All Repositories Cloned")