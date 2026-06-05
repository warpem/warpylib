---
name: release
description: Tag a new version and push to trigger the PyPI publish workflow. Usage: /release 1.2.3
disable-model-invocation: true
---

The user wants to release version $ARGUMENTS of warpylib to PyPI.

1. Verify the working tree is clean (`git status`). If there are uncommitted changes, stop and tell the user.
2. Verify you're on the `main` branch. If not, warn the user and ask if they want to proceed.
3. Confirm the version string looks valid (e.g. `1.2.3` or `1.2.3rc1`). Prefix with `v` to form the tag name.
4. Show the user the tag that will be created and pushed: `v$ARGUMENTS`
5. Ask the user to confirm before proceeding (this pushes to the remote and triggers PyPI publish).
6. If confirmed:
   - `git tag v$ARGUMENTS`
   - `git push origin v$ARGUMENTS`
7. Tell the user the GitHub Actions workflow will now build and publish to PyPI. They can monitor it at the Actions tab.
