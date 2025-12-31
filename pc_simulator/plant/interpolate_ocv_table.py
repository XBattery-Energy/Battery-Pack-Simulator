"""Interpolate OCV table values between key points to create smooth curve."""
import numpy as np

# Key points from real data
key_points = {
    0: 2.862,
    5: 3.112,
    10: 3.172,
    20: 3.279,
    50: 3.297,
    80: 3.314,
    90: 3.329,
    100: 3.472
}

# Create full table with interpolation
soc_values = list(range(101))
ocv_values = []

for soc in soc_values:
    if soc in key_points:
        ocv_values.append(key_points[soc])
    else:
        # Find surrounding key points
        lower_soc = max([k for k in key_points.keys() if k <= soc])
        upper_soc = min([k for k in key_points.keys() if k >= soc])
        
        if lower_soc == upper_soc:
            ocv_values.append(key_points[lower_soc])
        else:
            # Linear interpolation
            lower_ocv = key_points[lower_soc]
            upper_ocv = key_points[upper_soc]
            ratio = (soc - lower_soc) / (upper_soc - lower_soc)
            ocv = lower_ocv + ratio * (upper_ocv - lower_ocv)
            ocv_values.append(round(ocv, 3))

# Print in the format needed for cell_model.py
print("_OCV_SOC_TABLE_DISCHARGE = np.array([")
print("        # SOC%, OCV(V) - Interpolated from real data")
for i, (soc, ocv) in enumerate(zip(soc_values, ocv_values)):
    if i == 0:
        print(f"        [{soc:.1f}, {ocv:.3f}],   # {soc}% - fully discharged")
    elif i == 100:
        print(f"        [{soc:.1f}, {ocv:.3f}],  # {soc}% - fully charged")
    elif i % 10 == 0:
        print(f"        [{soc:.1f}, {ocv:.3f}],  # {soc}%")
    else:
        print(f"        [{soc:.1f}, {ocv:.3f}],")
print("    ])")

