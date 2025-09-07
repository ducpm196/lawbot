# Public test for the VLSP 2025 DRiLL: The challenge of Deep Retrieval in the expansive Legal Landscape
- The contents of the legal passages whose `id` is mentioned in the training are provided in `legal_corpus.json`.
- The data is provided in JSON format and UTF-8 encoding.
- Please refer to https://vlsp.org.vn/vlsp2025/eval/drill for more information. 

## File Overview and Structure

### `public_test.json`
- **Decription**: Public test containing only question text.  
- **Schema**:
  ```json
  {
    "qid": <integer>,             // Unique question identifier
    "question": "<string>",       // Text of the question
    "relevant_laws": []           // List of `aid` values (article IDs) from `legal_corpus.json`. This is the expected output.
  }
