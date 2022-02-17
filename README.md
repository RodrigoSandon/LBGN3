# Laboratory of Behavioral and Genomic Neuroscience

Supervisors: Dr. Patrick Piantadosi and Dr. Andrew Holmes

# TODO Streams

1. Decoding
2. Longitudinal Registration

# SUGGESTIONS

1. Fix x-axises of all figures that do't have the proper x axis?
2. Do significance analysis on the difference for the stacked plot?
3. Similarity map (DTW/Pearson corr) --> Decoders --> Interpretation Models?
4. Include Progressive Ratio Sessions into database?
5. BORIS Matlab to BORIS python for approach abort data?

# BUGS

1. I thought i fixed the truncation error, but it came back again? Need to make this fix simpler
   1. Just trim to 200 and that's it

# POSSIBLE BUGS

1. I ommitted any "NEW_SCOPE" substrings that might've been present in any session name, could this screw up the indv cell, between cell, and b/w mice alignment? I don't think so but we'll see.

# Main Project: ISX-ABET-SLEAP Pipeline

A library for extracting, preprocessing, aligning, and analyzing calcium activity data, behavioral data, and positional data using Inscopix (ISX), Animal Behavior Environment Test (ABET), and SLEAP (movement tracking) with 1-photon imaging data in mind.

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ISX-ABET-SLEAP.

```bash
pip install ISX-ABET-SLEAP
```

### Usage

```python
import ISX-ABET-SLEAP
```

### Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### License

[MIT](https://choosealicense.com/licenses/mit/)

# Subprojects:

1. Aligning Big Brains & Atlases (ABBA)
2. Merging Video Files
