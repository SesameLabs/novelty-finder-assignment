
## Installation

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the format detection with:

```bash
python novel_format_finder.py --ads_input ads.json --golden_output golden_output.json --output predictions.json
```

### Required Arguments

- `--ads_input`: Path to the JSON file containing the ad stream data
- `--output`: Path where the prediction results will be saved
- `--golden_output`: Path to the golden output file for validation


