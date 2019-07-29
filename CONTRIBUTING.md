Found a bug? Have a new feature to suggest? Want to contribute changes to the codebase? Make sure to read this first.

## Bug reporting

Your code doesn't work, and you have determined that the issue lies with `ivis`? Follow these steps to report a bug.

1. Your bug may already be fixed. Make sure to update to the current `ivis` master branch, as well as the latest stable TensorFlow distribution.

2. Search for similar issues. Make sure to delete `is:open` on the issue search to find solved tickets as well. It's possible somebody has encountered this bug already.

3. Make sure you provide us with useful information about your configuration: what OS are you using? What data types you are using?

4. Provide us with a script to reproduce the issue. This script should be runnable as-is and should not require external data download (use randomly generated data if you need to run `ivis` on some test data).

5. If possible, take a stab at fixing the bug yourself!

The more information you provide, the easier it is for us to validate that there is a bug and the faster we'll be able to take action.

---

## Requesting a Feature

You can also use Github issues to request features you would like to see in `ivis`, or changes in the `ivis` API.

1. Provide a clear and detailed explanation of the feature you want and why it's important to add.

2. Provide code snippets demonstrating the API you have in mind and illustrating the use cases of your feature.

3. After discussing the feature you may choose to attempt a Pull Request.


---

## Pull Requests

Here's a quick guide to submitting your improvements:

1. If your PR introduces a change in functionality, make sure you start by writing a design doc.

2. Make sure any new function or class you introduce has proper docstrings. Make sure any code you touch still has up-to-date docstrings and documentation.

4. Write tests. Your code should have full unit test coverage. If you want to see your PR merged promptly, this is crucial.

5. Run our test suite locally.

6. Make sure all tests are passing:    

7. We use PEP8 syntax conventions, but we aren't dogmatic when it comes to line length.


8. When committing, use appropriate, descriptive commit messages.

9. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

10. Submit your PR. If your changes have been approved in a previous discussion, and if you have complete (and passing) unit tests as well as proper docstrings/documentation, your PR is likely to be merged promptly.

---

## Adding new examples

Even if you don't contribute to the `ivis` source code, if you have an application of `ivis` that is concise and powerful, please consider adding it to our collection of examples!
