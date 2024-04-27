# Release Process

## 0. One-time Setup

To publish packages to PyPI and Test PyPI, you need to configure Poetry.
The following instructions are adapted from https://stackoverflow.com/a/72524326.

PyPI:

1. Get a token from https://pypi.org/manage/account/token/
2. Store the token:

```bash
poetry config pypi-token.pypi <token>
```

Test PyPI:

1. Add Test PyPI repository:

```bash
poetry config repositories.test-pypi https://test.pypi.org/legacy/
```

2. Get a token from https://test.pypi.org/manage/account/token/
3. Store the token:

```bash
poetry config pypi-token.test-pypi <token>
```

## 1. CLI Steps

Sourcetree has great built-in support for Git Flow from the `Repository > Git flow` menu. We list
the commands below, but this can also be done via the GUI if you prefer.

```sh
# Update working copy
# - Ensure your working copy is clean before starting (e.g. stash any WIP).
# - Fetch & pull latest origin/main
git checkout main && git pull

# (Optional) pre-commit
# - You shouldn't need to run pre-commit unless you've changed things manually
# - If you're seeing changes to poetry.lock, try clearing your Poetry cache & run again
poetry cache clear pypi --all
pre-commit run --all-files --hook-stage=manual

# Bump the version number
# - bump2version will update the version number, create a commit, and tag it with vx.y.z
bump2version patch # or minor or major

# Publish to PyPI:
# - Build and publish to PyPI
# - To publish to Test PyPI instead, use `poetry publish -r test-pypi`
poetry build
poetry publish

# Update remote repo
git push
git push origin --tags
```

## 2. GitHub Steps

-   Copy the **raw markdown** for the release notes in CHANGELOG: [https://github.com/obvslib/obvs/blob/main/CHANGELOG.md]
-   Once you've pushed the tag, you will see it on this page: [https://github.com/obvslib/obvs/tags]
-   Edit the tag and add the release notes
-   You will then see the release appear here: [https://github.com/obvslib/obvs/releases]
-   This also sends an email update to anyone on the team who has subscribed containing formatted release notes.
-   Once the release is created, edit the release and assign the milestone to the release. Save changes.

To finish, copy the release notes and post in any relevant Slack channel or email lists to inform members about the release.
