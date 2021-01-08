## Contributing Code
The preferred workflow for contributing is to clone the main repository on GitHub and develop on a branch. 

Steps:
- Clone the repo from your GitHub account to your local disk:
```git clone https://github.com/neurodata/brainlit.git```
- Create a feature branch to hold your development changes:
```git checkout -b my-feature```
- Develop the feature on your feature branch. Add changed files using git add and then git commit files:
```git add modified_files```
```git commit```
to record your changes in Git, push the changes to your branch with:
```git pull origin master```
```git push origin my-feature```

The repository is structured according to
![this model](https://nvie.com/img/git-model@2x.png)
([source](https://nvie.com/posts/a-successful-git-branching-model/))

## Pull Request Checklist
We recommended that your contribution complies with the following rules before you submit a pull request:

 - [ ] Give your pull request a helpful title that summarises what your contribution does. In some cases Fix <ISSUE TITLE> is enough. Fix #<ISSUE NUMBER> is not enough.
 - [ ] All public methods should have informative docstrings with sample usage presented as doctests when appropriate.
 - [ ] At least one paragraph of narrative documentation with links to references in the literature (with PDF links when possible) and the example.
 - [ ] All functions and classes must have unit tests. These should include, at the very least, type checking and ensuring correct computation/outputs.
 - [ ] Ensure all tests are passing locally using pytest. Install the necessary packages by:
```pip install pytest pytest-cov```
then run
```pytest```
or you can run pytest on a single test file by
```pytest path/to/test.py```
 - [ ] Run an autoformatter. We use black and would like for you to format all files using black. You can run the following lines to format your files.
```pip install black```
```black path/to/module.py```
