## Brief overview
This rule establishes guidelines for avoiding over-engineering solutions and ensuring that complex architectural decisions are made collaboratively with the user. The assistant should propose simple, direct solutions first and only consider more complex approaches when explicitly requested or when clearly necessary.

## Solution complexity protocol
- Always start with the simplest, most direct solution that addresses the immediate requirement
- Avoid adding unnecessary abstractions, patterns, or architectural complexity without user approval
- When multiple solution approaches are possible, present the simplest option first
- Only propose complex solutions when they are clearly necessary for the task requirements
- Prefer building the minimal set of executables or targets needed to validate changes

## When to ask for user input
- Before implementing complex design patterns (e.g., factory, observer, strategy)
- Before creating extensive abstraction layers or interfaces
- Before adding performance optimizations that increase code complexity
- Before implementing advanced architectural decisions
- When unsure whether a simpler solution would suffice

## Proposal format for complex solutions
- Present the simple solution first with clear explanation of why it's sufficient
- If proposing a complex solution, clearly justify why the simple approach won't work
- Provide 2-3 options ranging from simple to complex with trade-offs explained
- Ask the user to choose the appropriate level of complexity for their needs

## Examples of appropriate responses
- "I can solve this with a simple function, or we could implement a more complex pattern. The simple approach should work for your current needs. Would you like me to proceed with that, or would you prefer a more robust solution?"
- "Here's a straightforward implementation that addresses your requirement. If you need it to be more extensible or performant, we could add complexity, but this should work well for now."
- "I've identified three approaches: 1) Simple direct solution, 2) Modular approach with interfaces, 3) Full architectural pattern. The first should meet your needs. Which level of complexity would you prefer?"