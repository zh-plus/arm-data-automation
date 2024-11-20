detection_prompt = '''Please analyze the image and complete two tasks:

0. Translate the command to english
1. Provide a brief description focusing on the spatial relationship of key objects in the scene
2. From the given instruction, identify the start object (to be moved) and end object (destination)

Please output in this JSON format:
{
    "command": "Move the xxx to xxx."
    "description": "A clear description of the key objects' positions",
    "start": "object to be moved",
    "end": "destination object",
}

Example:
For instruction "Please help me put the red block on the house sketch"
{
    "command": "Move the red block to house sketch."
    "description": "The house sketch is at the top, and the red block is at the bottom",
    "start": "red block",
    "end": "house sketch",
}

Only return the JSON object, no additional text.'''

judge_prompt = '''Evaluate whether the robotic arm has successfully completed its assigned task.

Input will contain:
1. Initial scene description
2. Command given to the robot
3. Final image after task execution

Please analyze whether the final image shows successful completion of the commanded action.
Return only "Yes" or "No" as your answer.

Example:
Input:
- Initial description: "The house sketch is at the top, and the red block is at the bottom"
- Command: "Please help me put the red block on the house sketch"
- Final image: <IMG>

Output: Yes'''