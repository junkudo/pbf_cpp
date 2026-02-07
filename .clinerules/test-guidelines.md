## Brief overview
This rule establishes guidelines for writing comprehensive tests for new interfaces and functionality in the PBF project.

## Test coverage requirements
- Write tests for all new public interfaces and functions
- Include both unit tests and integration tests where appropriate
- Test edge cases and boundary conditions
- Verify error handling and invalid input scenarios

## Test organization
- Place unit tests in the tests/ directory with descriptive names
- Use fixture classes for shared test data and setup
- Group related tests using appropriate test suites
- Follow existing naming conventions (e.g., *_tests.cpp)

## Test quality standards
- Tests should be independent and not rely on external state
- Use clear, descriptive test names that explain the scenario
- Include assertions with meaningful error messages
- Mock external dependencies when testing isolated components

## Testing framework
- Use Google Test (gtest) framework for C++ tests
- Follow established patterns from existing test files
- Use appropriate assertion macros (EXPECT_EQ, ASSERT_TRUE, etc.)
- Implement custom matchers for complex validation when needed

## Running tests after changes
- After making code or test changes, run the relevant test suite before marking the task complete
- Always rebuild before running tests (e.g., `cmake --build build`) to ensure binaries are up to date
- Prefer `ctest --test-dir build --output-on-failure` unless a narrower test target is more appropriate

## Performance considerations
- Include performance benchmarks for critical algorithms
- Test with realistic data sizes and scenarios
- Monitor test execution time to prevent regression
- Use parameterized tests for testing multiple scenarios efficiently