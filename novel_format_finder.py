import json
import argparse
import openai
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Set your OpenAI API key
openai.api_key = "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"

# Basic implementation of novelty detection using OpenAI GPT-4 Vision
def detect_novelty(ads_list):
    predictions = []
    known_formats = []
    known_formats_descriptions = ""

    for ad in ads_list:
        prompt = """
        You are an expert creative strategist with deep expertise in classifying visual ad formats. Your task is to analyze each ad image provided and determine whether it introduces a novel visual format or matches an existing format already known.

        A visual format is defined strictly by visual characteristics and must NOT rely on advertiser-specific imagery or branding elements. Formats fall into exactly one of these categories:

        Structural Layout / Arrangement-Based Format:
        - Defined by arrangement of elements (e.g., Split screen, Multi-product grid, Framing, Feature tags with arrows, Bottom headline overlay, Centered text with product backdrop).

        UI / App Mimicry:
        - Mimics common digital interfaces or applications (e.g., Chat bubbles, Notification popups, Social media feed, Message threads, App store listings).

        Physical / Real-world Mimicry:
        - Mimics real-life objects or surfaces (e.g., Post-It notes, Torn paper, Billboards, Signboards, Newspaper layout, Magazine covers).

        Meme:
        - Uses recognizable meme structures or templates (e.g., Drake Reaction Meme, Distracted Boyfriend, Expanding Brain, 'This is fine' dog meme).

        Graphic Design Style:
        - Distinct artistic or graphic styles (e.g., Pop Art, Watercolor, Retro Illustration, Comic book style, Minimalist flat design).

        Familiar Conceptual Structures:
        - Common conceptual layouts or patterns widely recognized (e.g., Venn Diagrams, Quizzes, Tic Tac Toe grids, Scrabble boards, Checklists, Before-and-after comparisons).

        Important Considerations for Naming Formats:
        - Clearly reference the primary defining visual attribute.

        For Structural Layout, include the dominant content type (headline, lifestyle shot, studio shot, product close-up, offer badge) but NOT specific details (e.g., do NOT specify 'woman on a beach').
        In general format name and format description not never be speicifc to the content type or advertiser-specific content.
        In terms of permissible dominant content types that deserve contributing to format name and format description, only include the following:
        - Lifestyle shot
        - Studio shot
        - Product close-up
        - Feature tag
        - User-generated content (UGC)
        - Before and after images
        - Testimonials
        - Icons or graphics or illustrations
        

        Examples:

        Correct: 'Lifestyle shot with bottom headline overlay'

        Incorrect: 'Woman on beach lifestyle shot'

        Existing Formats:
        You will be provided a list of existing format names and descriptions. If the analyzed ad matches one of these formats, set "isNewFormat" to false, and return the existing format's name and description. Otherwise, set "isNewFormat" to true and provide a new, clear format name and a detailed description.

        Your Output Format (STRICTLY adhere to this JSON format):
        {
        "isNewFormat": true/false,
        "formatName": "Clear and descriptive format name based on rules above",
        "formatDescription": "Detailed description of the visual characteristics defining this format."
        }

        Your response must be concise, accurate, and strictly follow the guidelines above to ensure the correct classification of visual ad formats.
        """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt + "\n\nExisting Formats and Descriptions:\n" + known_formats_descriptions},
                    {"type": "image_url", "image_url": {"url": ad['imageUrl']}}
                ]
            }],
            response_format={"type": "json_object"}
        )

        try:
            prediction = json.loads(response.choices[0].message.content)
            print("\nPrediction for Ad ID:", ad['adId'])
            print("Image URL:", ad['imageUrl'])
            print("Format Details:")
            print(json.dumps(prediction, indent=2))
            print("-" * 80)
        except Exception as e:  # Catch any errors during parsing
            print(f"Parsing error: {e}")
            prediction = {"isNewFormat": False, "formatName": "Unknown Format"}

        predictions.append({
            "adId": ad['adId'],
            "isNewFormat": prediction["isNewFormat"],
            "formatName": prediction["formatName"]
        })

        # Update known formats if a new format is found
        if prediction["isNewFormat"]:
            new_format = {
                "name": prediction["formatName"],
                "description": prediction["formatDescription"]
            }
            known_formats.append(new_format)
            
            # Rebuild the formats description string
            known_formats_descriptions = "\n".join([
                f"- {format['name']}: {format['description']}" 
                for format in known_formats
            ])
    return predictions

# Load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Save JSON data to a file
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Evaluate predictions against ground truth
def evaluate_predictions(predictions, known_formats):
    y_true = []
    y_pred = []
    misclassified = []

    # For each prediction, check if its ad ID is in known formats
    for i, pred in enumerate(predictions):
        pred_is_new = pred['isNewFormat']  # What model predicted
        true_is_new = known_formats[i]['isNewFormat']  # Ground truth

        y_true.append(true_is_new)
        y_pred.append(pred_is_new)

        if true_is_new != pred_is_new:
            misclassified.append({
                "adId": pred["adId"],
                "true": true_is_new,
                "predicted": pred_is_new
            })

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not New Format", "New Format"]))

    print(f"Precision (New Format): {precision:.2f}")
    print(f"Recall (New Format): {recall:.2f}")
    print(f"F1 Score (New Format): {f1_score:.2f}")

    if misclassified:
        print("Misclassified Ads:")
        for mis in misclassified:
            print(f"AdId: {mis['adId']}, True: {mis['true']}, Predicted: {mis['predicted']}")

# Main function to run the entire pipeline
def main(args):
    ads_stream = load_json(args.ads_input)
    golden_ads = load_json(args.golden_output)

    predictions = detect_novelty(ads_stream)
    save_json(predictions, args.output)
    predictions = load_json(args.output)
    evaluate_predictions(predictions, golden_ads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Novel Format Finder Evaluation Script')
    parser.add_argument('--ads_input', required=True, help='Path to input ads JSON file')
    parser.add_argument('--golden_output', required=True, help='Path to golden formats JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSON file for predictions')

    args = parser.parse_args()
    main(args)
