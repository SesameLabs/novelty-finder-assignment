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
        Analyze this ad image and determine if it represents a new visual format or matches an existing format.

        A format is defined by how the visual elements are arranged (like split screen, grid layout, etc).

        Review the list of existing formats below carefully. For your response:
        1. If the ad matches ANY existing format:
           - Set isNewFormat to false
           - Use the matching format's name as formatName
           - Use the matching format's description as formatDescription
        2. If the ad doesn't match any existing format:
           - Set isNewFormat to true
           - Create a new descriptive formatName
           - Provide a clear formatDescription

        Provide your response in JSON format:
        {
        "isNewFormat": true/false,
        "formatName": "Name of the format",
        "formatDescription": "Description of the visual format"
        }

        Existing Formats:
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
