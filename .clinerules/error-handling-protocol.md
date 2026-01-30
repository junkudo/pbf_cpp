## Brief overview
This rule establishes guidelines for handling test failures and unexpected errors during development. When errors occur, the assistant should proactively propose solutions and next steps to the user rather than waiting for explicit instructions.

## Error response protocol
- When test failures or unexpected errors are encountered, immediately analyze the issue and propose 2-3 potential solutions
- Present proposals in a clear, structured format with brief explanations of each option
- Include both immediate fixes and longer-term improvements when appropriate
- Ask the user to choose from the proposed solutions rather than asking open-ended questions

## Proposal format
- Use numbered or bullet-point format for multiple options
- Keep proposals concise but informative
- Include expected outcomes and any trade-offs for each option
- Prioritize solutions from most direct fix to more comprehensive approaches

## When to apply
- Test failures during development or CI/CD pipeline
- Compilation errors that block progress
- Runtime errors in development environment
- Unexpected behavior in implemented features
- Performance issues that impact functionality

## Examples of appropriate responses
- "I've identified the test failure. Here are three potential solutions: 1) Fix the null pointer issue in the initialization logic, 2) Add proper error handling for edge cases, 3) Update the test expectations to match the new behavior"
- "The compilation error appears to be related to missing dependencies. I suggest: 1) Installing the required package, 2) Updating the build configuration, or 3) Using an alternative implementation"