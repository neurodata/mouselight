## Contributing Code
The preferred workflow for contributing is to clone the main repository on GitHub and develop on a branch. 

Steps:
- Clone the repo from your GitHub account to your local disk:
```git clone https://github.com/neurodata/brainlit.git```
- Create a feature branch to hold your development changes:
```git checkout -b <my-feature>```
- Develop the feature on your feature branch. Add changed files using git add and then git commit files:
```git add <modified_files>```
```git commit```
to record your changes in Git, push the changes to your branch with:
```git pull origin develop```
```git push origin <my-feature>```

The repository is structured according to
![this model](https://nvie.com/img/git-model@2x.png)
([source](https://nvie.com/posts/a-successful-git-branching-model/))

## Pull Request Checklist
We recommended that your contribution complies with the following rules before you submit a pull request:

 - [ ] Give your pull request a helpful title that summarises what your contribution does. In some cases Fix <ISSUE TITLE> is enough. Fix #<ISSUE NUMBER> is not enough.
 - [ ] All public methods should have informative docstrings with sample usage presented as doctests when appropriate.
 - [ ] All major functions and classes should have at least one paragraph of narrative documentation with links to references in the literature (with PDF links when possible) and the example.
 - [ ] All functions and classes must have pytest unit tests. These should include, at the very least, type checking and ensuring correct computation/outputs. Test files should be located in a folder next to the files they are testing. Reminder that just because the code is covered does not mean it is tested. For example, code that is tested on a single type of s3 data may not work on other types of data.
 - [ ] Ensure all tests are passing locally using pytest. Install the necessary packages by:
```pip install pytest pytest-cov```
then run
```pytest```
or you can run pytest on a single test file by
```pytest path/to/test.py```
 - [ ] **Run an autoformatter.** We use black and would like for you to format all files using black. You can run the following lines to format your files.
```pip install black```
```black path/to/module.py```
 - [ ] Ensure that your jupyter notebooks run successfully when you restart the kernel and run all cells. They should be located in [docs/notebooks](https://github.com/neurodata/brainlit/tree/develop/docs/notebooks).
 - [ ] You are responsible for updating our [documentation page](https://brainlit.netlify.app/) according to your changes. This may involve updating `.rst` files for core code and notebooks:
   - [ ] For core code edit [index.rst](https://github.com/neurodata/brainlit/blob/develop/docs/reference/index.rst) and your submodule's `.rst` file such as [algs.rst](https://github.com/neurodata/brainlit/blob/develop/docs/reference/algs.rst).
   - [ ] For notebooks, edit [tutorial.rst](https://github.com/neurodata/brainlit/blob/develop/docs/tutorial.rst).
   - [ ] Run `./docs/build_docs.sh` from brainlit's root directory. Output will be in `docs/_build/`. You should open `docs/_build/html/index.html` to preview the new documentation.
   - [ ] After your PR is merged, please (if appropriate) close any relevant issues and delete any relevant branches.
