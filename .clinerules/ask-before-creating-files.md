## Brief overview
This rule establishes guidelines for seeking user approval before creating new files or deleting existing files in the PBF project. This ensures collaborative development and prevents unwanted changes to the codebase.

## File creation protocol
- Always ask the user for approval before creating new files
- Use the ask_followup_question tool to request permission
- Provide clear justification for why the new file is needed
- Include the proposed filename and location in the request
- Wait for explicit user confirmation before proceeding with file creation

## File deletion protocol
- Always ask the user for approval before deleting existing files
- Use the ask_followup_question tool to request permission
- Provide clear justification for why the file deletion is necessary
- Include the filename and reason for deletion in the request
- Wait for explicit user confirmation before proceeding with file deletion

## Alternative approaches
- Consider modifying existing files instead of creating new ones when possible
- Explore refactoring options before suggesting file deletions
- Propose incremental changes that can be reviewed step by step
- Suggest temporary or experimental files when testing new functionality

## Communication requirements
- Clearly explain the purpose and benefits of any proposed file changes
- Provide context about how the changes fit into the overall project architecture
- Be prepared to discuss alternatives if the user has concerns
- Document the reasoning behind file creation/deletion decisions