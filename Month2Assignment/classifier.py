import pandas as pd
import json
 
from litellm import completion
from dotenv import load_dotenv
 
# .env file load karega jahan GROQ_API_KEY stored hai
load_dotenv()
 
def create_prompt(claim_text):
 
    return f"""
You are an insurance claim classifier.
 
Classify the claim into:
 
1. claim_type
   - Motor
   - Property
   - Liability
 
2. tone
   - Calm
   - Frustrated
   - Urgent
 
3. legal_action
   - Yes
   - No
 
Return ONLY valid JSON in this format:
{{
  "claim_type": "Motor",
  "tone": "Calm",
  "legal_action": "No"
}}
 
Claim:
{claim_text}
"""
 
def classify_claim(claim_text):
 
    prompt = create_prompt(claim_text)
 
    response = completion(
        model="groq/llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
 
    return response.choices[0].message.content
 
# CSV Read Karna    
df = pd.read_csv("Claim_description.csv")
 
# Results store karne ke liye list
results = []
 
# Har claim par loop
for index, row in df.iterrows():
 
    claim_text = row["description"]
 
    print(f"\nProcessing Claim {row['claim_id']}...")
 
    try:
    # LLM Response
        result = classify_claim(claim_text)
 
        print(result)
 
     # String JSON ko Python Dictionary me convert karna
        classification = json.loads(result)
 
    except Exception as e:
 
        print(f"Error processing claim {row['claim_id']}: {e}")
 
        classification = {
            "claim_type": "Unknown",
            "tone": "Unknown",
            "legal_action": "Unknown"
        }
 
    results.append({
        "claim_id": row["claim_id"],
        "description": claim_text,
        "classification": classification
    })
 
 
# JSON file save karna
with open("output.json", "w") as file:
    json.dump(results, file, indent=4)
 
print("\nAll claims processed successfully!")
print("Results saved in output.json")
 