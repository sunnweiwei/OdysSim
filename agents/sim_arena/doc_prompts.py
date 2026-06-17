# Copyright 2025 Individual Contributor: OdysSim Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DOC_SIMULATOR_SYSTEM_PROMPT = """You are an AI assistant tasked with role-playing as a user seeking help from an AI writing assistant to create a document. Your primary goal is to accurately simulate a user with the specific characteristics defined in the profile below. This profile simulation is crucial for maintaining authenticity in the conversation.

# Writing Objectives
- Document Type: {document_type}
- Document Goal: {intent}
- Document Length: Between 100 and 500 words

# Pre-writing Materials
{pre_writing_materials}

*Note: Pre-writing materials are the factual or contextual notes and ideas the user has prepared before engaging with the assistant.*

# User Profile:
{user_profile}

# Guidelines for Your Role as a User:
1. Act according to the provided user profile, with the overall goal of creating a well-written document with the AI writing assistant.
2. Each message can involve asking questions, giving instructions, offering feedback, suggesting changes, correcting inaccuracies, or refining the draft, in a way a real user might.
3. Express concerns, preferences, and factual corrections naturally.
4. Share information and pre-writing materials gradually, as you would in a natural conversation, rather than providing everything at once.
5. Maintain the profile characteristics consistently across the conversation.

### Maintaining Profile Characteristics:
- How to express your thoughts according to the given profile
- Which profile characteristics are most relevant to this response
- How to naturally incorporate these characteristics into your message

# Output Format
- First user turn: output only `Message: ...`
- Later user turns: output only `Message: ...`
- If you are satisfied with the document or the conversation is no longer productive: output exactly `terminate conversation`

# Termination
Output `terminate conversation` if any is true:
1) You are satisfied with the final document and have no further revisions.
2) The conversation is no longer productive (e.g., going in circles, not addressing your needs, or the assistant's messages are unhelpful).

# Notes
- Do not dump all information at once unless it is natural to do so.
- Continue to behave like the same user throughout the conversation.
- Stay focused on improving the document toward your intended goal.

Stay in character as the specified user throughout your output, following the guidelines and user profile carefully."""


DOC_SIMULATOR_INITIAL_USER_MESSAGE_TEMPLATE = """Document Type:
{document_type}

Document Goal:
{intent}

Pre-writing Materials:
{pre_writing_materials}

Conversation starts now. As the user, send your first message to the writing assistant.
Output format: Message: [your initial message]"""


ASSISTANT_SYSTEM_PROMPT_DOC = (
    "You are a helpful AI writing assistant. Your goal is to collaborate with the user to create "
    "and refine a document that matches their goals, factual background, preferences, and style. "
    "Be responsive to feedback, revise drafts faithfully, and help the user iteratively improve the document. "
    "Keep the document concise: the final document must be between 100 and 500 words."
)


ASSISTANT_FIRST_TURN_USER_TEMPLATE_DOC = (
    "Here is the document creation task that you will help me with:\n"
    "Document Type: {document_type}\n"
    "Document Goal: {intent}\n"
    "Pre-writing Materials:\n{pre_writing_materials}\n\n"
    "{message}"
)


TERMINATION_PROMPT_TEMPLATE_DOC = """You are given a sequence of **User Messages** from a **document creation** conversation, along with the **Document Type** and **Document Goal**. The user is collaborating with an AI writing assistant to produce and refine a document. Your task is to determine the optimal point to end the conversation based on the user's progress and satisfaction.

## Input Format
### Document Type
{document_type}

### Document Goal
{intent}

### User Messages
{user_messages}

## Termination Criteria
End the conversation when **ANY** of these occur:
1. **Final Satisfaction**: The user is satisfied with the final document and has no further revisions.
2. **Unproductive Conversation**: The conversation is no longer productive (e.g., going in circles, not addressing the user's needs, or the assistant's messages are unhelpful).

## Output Format
```json
{{
    "Analysis": [
        "Turn 1: [Brief analysis of user's requests/feedback]",
        "Turn 2: [Brief analysis of user's requests/feedback]",
        ...
    ],
    "Ending Turn Number": X,
    "Termination Reason": "[One of the two criteria above]"
}}
Notes:
1. The "Ending Turn Number" should be the last turn that is relevant to achieving a finalized, satisfactory document.
2. End the conversation if the user explicitly indicates no further changes are needed (Criterion 1) or if it becomes clear that no productive progress is happening (Criterion 2)."""


DOC_SIMULATOR_WRITING_STYLE_FEATURES_TEXT = """- Frequency of Grammatical Errors: How often does the user break basic grammar rules?
- Sentence Complexity: Does the user primarily use simple sentences, or do they also use compound and complex structures?
- Spelling Consistency: Does the user often misspell words or make typos?
- Punctuation Usage: How does the user employ punctuation? Are marks often missing or excessive?
- Capitalization Patterns: Does the user consistently capitalize letters correctly, or do they use all lowercase or random capitalization?
- Range of Words: Does the user stick to basic vocabulary or incorporate a broader lexicon?
- Repetitive or Filler Words: Does the user rely heavily on filler terms or repeat the same words and phrases often?
- Level of Formality: Is the user's language generally formal, casual, or somewhere in between?
- Use of Slang, Contractions, or Emojis: Does the user employ slang, contractions, emoticons, or emojis, and how frequently?
- Ambiguous or Clear Language: Is the user's message easy to interpret, or does it contain incomplete or ambiguous phrasing?
- Fragmentation of Sentences: Does the user often use fragmented sentences or complete sentences with clear structure?
- Complexity of Requests: How complex are the user's requests? Do they often involve multiple steps or detailed instructions?
- Clause Variety: How varied are the user's clauses within sentences?
- Politeness Frequency: How frequently does the user use politeness markers such as thank you, please, or could you?
- Sentence Initiation Variety: Does the user start sentences in varied ways, or do they follow a repetitive pattern?"""


DOC_SIMULATOR_INTERACTION_STYLE_FEATURES_TEXT = """- Message Length: What is the range of the length of the user's queries?
- Answer Thoroughness: When responding to questions, does the user tend to give detailed, complete responses or brief, minimal answers?
- Information and Clarification Seeking: How frequently does the user ask for examples, additional explanations, or factual clarification?
- Acknowledgment and Feedback Style: How does the user acknowledge understanding or receipt of information? Does the user provide ongoing verbal feedback?
- Context and Coherence: Does the user refer back to previous messages or maintain conversation context and coherence?
- Adaptability and Feedback Implementation: How does the user respond to and implement suggestions or corrections?
- Emotional Expression: How does the user communicate frustration, enthusiasm, satisfaction, or other emotions?
- Persistence and Redundancy in Feedback: Does the user request the same type of feedback repeatedly without significant changes?
- Personalization and Creative Engagement: Does the user incorporate personal insights, creative suggestions, and specific experiences into their messages?
- Iterative and Incremental Refinement: Does the user refine the content incrementally across multiple turns?
- Specificity and Goal Orientation in Feedback: How specific and goal-oriented are the user's feedback and modification requests?
- Balance of Instruction and Inquiry: Does the user balance between giving specific instructions and asking for suggestions or ideas from the AI?
- Structured and Methodical Feedback: Does the user follow a highly structured and methodical approach in their feedback?"""


EVAL_SIMULATOR_WRITING_STYLE_LIKERT_PROMPT_TEMPLATE_DOC = """You are an expert in analyzing document creation conversations. Your task is to evaluate how similar a simulated user's writing style is to a real user's writing style in their queries to an AI writing assistant.

# Input
## Task
Document creation

## Document Type
{document_type}

## Document Goal
{intent}

## Real User Queries
{real_user_queries}

## Simulated User Queries
{simulated_queries}

# Analysis Features
Compare the following writing style features between the real user's queries and the simulated user's queries to guide your evaluation:
{features}

## Similarity Rating Scale
Rate the overall writing style similarity on a scale of 1-5:
1: Simulated user's writing style is completely different from real user's writing style
2: Simulated user's writing style shows significant differences from real user's writing style
3: Simulated user's writing style shows notable differences while maintaining some similarity
4: Simulated user's writing style is very similar to real user's with minor differences
5: Simulated user's writing style is nearly indistinguishable from real user's writing style

# Output Format
## Feature Analysis:
- Analyze the listed features.
- Compare how the real user and the simulated user differ or align for those features.

Results (JSON):
{{
"key_differences": ["list specific differences in writing style here"],
"similarity_score": "1-5"
}}

# Note
- Your goal is to determine how closely the simulated queries match real user queries in writing style.
- Focus on the user queries only, not the writing assistant's responses."""


EVAL_SIMULATOR_INTERACTION_STYLE_LIKERT_PROMPT_TEMPLATE_DOC = """You are an expert in analyzing document creation conversations. Your task is to evaluate how similar a simulated user's interaction style is to a real user's interaction style in their conversation with an AI writing assistant.

# Input
## Task
Document creation

## Document Type
{document_type}

## Document Goal
{intent}

## Conversation between Real User and AI Assistant
{real_conversation}

## Conversation between Simulated User and AI Assistant
{simulated_conversation}

# Analysis Features
Compare the following interaction style features between the real user's queries and the simulated user's queries to guide your evaluation:
{features}

## Similarity Rating Scale
Rate the overall interaction style similarity on a scale of 1-5:
1: Simulated user's interaction style is completely different from real user's interaction style
2: Simulated user's interaction style shows significant differences from real user's interaction style
3: Simulated user's interaction style shows notable differences while maintaining some similarity
4: Simulated user's interaction style is very similar to real user's with minor differences
5: Simulated user's interaction style is nearly indistinguishable from real user's interaction style

# Output Format
## Feature Analysis:
- Analyze the listed features.
- Compare how the real user and the simulated user differ or align for those features.

## Results (JSON):
{{
"key_differences": ["list specific differences in interaction style here"],
"similarity_score": "1-5"
}}

# Note
- Focus on user interaction style, not writing assistant quality.
- Your goal is to determine how closely the simulated user matches the real user in interaction style."""


EVAL_SIMULATOR_TURING_TEST_PROMPT_TEMPLATE_DOC = """You are an expert in conversation analysis and AI detection. You will compare two document creation conversations, each featuring a user interacting with an AI writing assistant. One user is a real human and the other is an AI user simulator. Your goal is to determine which conversation features the human user and provide a single confidence rating for your decision.

## Task
Document creation

## Document Type
{document_type}

## Document Goal
{intent}

## Conversations
<Conversation 1>
{conversation_1}
</Conversation 1>

<Conversation 2>
{conversation_2}
</Conversation 2>

## Output Format
Provide a detailed analysis of both conversations, focusing on indicators of human or AI behavior. Then make a single determination of which conversation features the human user and how confident you are in that judgment.

Structure your response as follows:
1. Analysis of Conversation 1
- Provide an analysis of the first conversation, noting key indicators of human or AI behavior.

2. Analysis of Conversation 2
- Provide an analysis of the second conversation, noting key indicators of human or AI behavior.

3. Comparison and Reasoning
- Compare the two conversations, highlighting the main differences and similarities that inform your decision.

4. Decision
Provide your decision using the following JSON format:
{{
"conversation_with_human_user": "1 or 2",
"confidence_rating": [percentage between 0-100]
}}

## Factors you can consider in your analysis
1. Language Use in Queries - Does the user's phrasing sound natural and varied, or is it overly formal, structured, or robotic?
2. Contextual Awareness - Does the user adapt based on previous suggestions, incorporating feedback in a flexible way, or do they rigidly follow patterns?
3. Variation in Requests - Does the user explore different approaches, styles, tones, or revisions naturally, or behave predictably and systematically?
4. Engagement and Exploration - Does the user ask exploratory questions, offer nuanced preferences, or only issue direct mechanical commands?
5. Interaction Flow - Does the conversation feel fluid and dynamic, with natural clarifications, refinements, and side discussions, or does it follow an overly structured pattern?

## Note
- Focus primarily on the user's messages rather than the writing assistant's.
- Be aware that a sophisticated AI might mimic human behavior convincingly, so look for subtle hints.
- First output your analysis and then the final decision in JSON.
"""


EVAL_SIMULATOR_ATTRIBUTE_FULFILLMENT_PROMPT_TEMPLATE_DOC = """You are an expert in communication analysis and AI interaction evaluation. Your task is to analyze the user messages and determine whether they match the provided feature description.

# Feature Description
{feature_description}

# Conversation
{conversation_text}

# Binary Classification Criteria
Match - The user's {feature_category} matches the feature description.
Definition: The user's messages demonstrate the characteristics described in the feature description. Their communication pattern aligns with what the description outlines.

No Match - The user's {feature_category} does not match the feature description.
Definition: The user's messages do not demonstrate the characteristics described in the feature description. Their communication pattern differs from what the description outlines.

# Classification Guidelines
- Focus exclusively on the user messages, not the AI writing assistant responses.
- Compare the user's {feature_category} directly to the feature description.
- Consider the overall pattern across all messages, not just isolated instances.

# Output Format
* Analysis: [Provide a thorough analysis comparing the user's {feature_category} to the feature description, citing specific examples and describing the overall pattern]
* Classification: [State either "Match" or "No Match"]
"""


EVAL_SIMULATOR_ALL_ATTRIBUTES_FULFILLMENT_PROMPT_TEMPLATE_DOC = """You are an expert in communication analysis and AI interaction evaluation. Your task is to analyze the user messages in the conversation below and determine whether they match each of the provided feature descriptions.

# Conversation
{conversation_text}

# Features to Evaluate
{features_text}

# Binary Classification Criteria
For each feature:
Match - The user's messages demonstrate the characteristics described in the feature description.
No Match - The user's messages do not demonstrate the characteristics described in the feature description.

# Classification Guidelines
- Focus exclusively on the user messages, not the AI writing assistant responses.
- Consider the overall pattern across all messages, not just isolated instances.

# Output Format
Return a JSON array where each element corresponds to one feature in the same order as listed above:
{{
  "results": [
    {{"feature_name": "<name>", "analysis": "<brief analysis>", "classification": "Match or No Match"}},
    ...
  ]
}}"""


EVAL_ASSISTANT_INTERACTION_RATING_PROMPT_TEMPLATE_DOC = """You are an expert in writing collaboration and AI writing assistant evaluation. Your task is to analyze a conversation between a user and an AI writing assistant about creating a document, then rate the AI writing assistant's performance on a scale of 1 to 10 based on the criteria below.

# Input
<Conversation>
{conversation}
</Conversation>

# Rating Criteria:
Score 1 ~ 2 (Very Poor):
The AI writing assistant consistently failed to understand the user's inputs, provided irrelevant or nonsensical responses, and made the interaction frustrating and unproductive.

Score 3 ~ 4 (Poor):
The AI writing assistant frequently misunderstood the user's requests, offered minimal assistance that didn't address the user's needs, and required repeated clarifications.

Score 5 ~ 6 (Average):
The AI writing assistant was somewhat helpful but had noticeable issues with comprehension or responsiveness, providing partially relevant responses that contained errors or omissions.

Score 7 ~ 9 (Good):
The AI assistant generally understood the user's needs and provided relevant, helpful responses.
There may be some issues with clarity, completeness, or minor mistakes, but overall it meets the user's primary objectives for document creation.
7: Helpful but has at least one moderate shortcoming.
8: Addresses the user's needs effectively with only a few minor or sporadic errors.
9: Near-excellent, with negligible gaps; it exceeds the user's expectations in most ways but still has a small area for improvement.

Score 10 (Very Good):
The AI writing assistant's performance is excellent, with effectively no significant flaws.
The user is highly satisfied with clarity, depth, and relevance throughout the conversation.

# Note:
1. Focus on the AI writing assistant's responses and how effectively it assists the user with document creation.
2. Use the user's feedback and questions as a gauge to assess the assistant's helpfulness, clarity, and responsiveness.
3. Provide specific analysis referencing the conversation to support your evaluation.

# Output Format:
* Analysis: [Provide a thorough analysis of the AI writing assistant's performance, considering the criteria above]
* Strengths: [List the key strengths demonstrated by the AI writing assistant in the conversation]
* Areas for Improvement: [Identify any issues or weaknesses in the assistant's performance]
* Rating: [Provide a single numeric rating between 1 and 10]"""


EVAL_DOCUMENT_EXTRACTION_PROMPT_TEMPLATE_DOC = """You are a document finalizer. Your task is to extract the final version of a document from a conversation between a user and an AI writing assistant.

# Input
<Conversation>
{conversation}
</Conversation>

# Instructions
1. Carefully read the entire conversation to identify every modification made to the document.
2. Combine all the modifications in the order they were made to determine the final version of the document.
3. Output only the final document content. Do not include any user queries, model responses, or conversational commentary.
4. If no document content exists or the final document is empty, output an empty string for the document content.

# Output Format
You must output in the following JSON format:
```json
{{
  "Thought": "Provide an analysis explaining whether a document was created and, if so, describe the document creation process throughout the conversation.",
  "Final Document": "Final document content, use empty string if the document is empty."
}}
```"""


EVAL_DOCUMENT_RATING_PROMPT_TEMPLATE_DOC = """You are an expert in writing collaboration and AI writing assistant evaluation. Your task is to analyze the final document produced by the AI writing assistant, then rate it on a scale of 1 to 10 based on the criteria below.

# Input
## Writing Objectives
- Document Type: {document_type}
- Document Goal: {intent}
- Document Length: Between 100 and 500 words

## Document Preferences
{document_preferences}

## Final Document
{final_document}

# Rating Criteria
Score 1-2 (Very Poor):
The document contains numerous errors, inaccuracies, or irrelevant content, lacks coherence and structure, and is unusable for the user's needs.

Score 3-4 (Poor):
The document has significant issues such as incomplete sections, misleading information, or poor organization, only partially addresses the user's instructions, and requires substantial revisions.

Score 5-6 (Average):
The document meets basic requirements but includes noticeable errors or omissions, provides some useful content but lacks depth or clarity, and requires moderate revisions to improve quality.

Score 7-8 (Good):
The document is well-organized, covers the key topics as instructed, contains accurate and relevant information with minor errors, and serves as a strong foundation that fulfills the user's main needs.

Score 9-10 (Very Good):
The document is comprehensive, insightful, and meticulously crafted, exceeds expectations by providing exceptional clarity and depth, requires minimal to no revisions, and significantly achieves the user's needs.

# Note
1. Focus on the final document's clarity, completeness, correctness, and relevance to the user's needs.
2. Provide specific analysis referencing the document to support your evaluation.

# Output Format
* Analysis: [Provide a thorough analysis of the final document's quality, referencing the criteria above]
* Strengths: [List the key strengths in the final document]
* Areas for Improvement: [Identify any issues or weaknesses in the final document]
* Rating: [Provide a single numeric rating between 1 and 10]"""
