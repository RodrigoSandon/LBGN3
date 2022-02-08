"""
The goal here is to:
1) Correct time points for velocity with GPIO (same addition as I did for the ABET file)
    - a time value from ABET_GPIO-processed - ABET_processed = GPIO correction
    - add this GPIO correction to all the times in the speed file
2) Downsample the file
    - Every 3 (after including the first one), extract those rows and make a new csv file

"""

