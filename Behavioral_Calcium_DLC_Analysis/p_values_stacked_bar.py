import numpy as np
from scipy.stats import chi2_contingency

def perform_chi_square_test(session_dicts):
    keys = ['+', '-', 'Neutral']
    observed = np.array([[len(d[key]) for key in keys] for d in session_dicts])
    chi2, p, _, _ = chi2_contingency(observed)
    return p

#Pre RDT RM Session

pre_rdt_rm={
    "mannwhitneyu_Block_Choice_Time_s_1dot0_57_minus10_to_minus5_1_to_4" : {'+': ['1_C06', '1_C11'], '-': ['1_C21', '1_C23', '1_C26', '3_C04', '3_C05', '3_C07'], 'Neutral': ['1_C08', '1_C09', '1_C19', '1_C27', '3_C01', '3_C03', '3_C09', '3_C11', '3_C12', '6_C05', '6_C08', '6_C10']},
"mannwhitneyu_Block_Choice_Time_s_2dot0_57_minus10_to_minus5_1_to_4" : {'+': ['1_C06', '1_C11', '1_C19'], '-': ['1_C21', '1_C26', '3_C04', '3_C05'], 'Neutral': ['1_C08', '1_C09', '1_C23', '1_C27', '3_C01', '3_C03', '3_C07', '3_C09', '3_C11', '3_C12', '6_C05', '6_C08', '6_C10']},
"mannwhitneyu_Block_Choice_Time_s_3dot0_57_minus10_to_minus5_1_to_4" : {'+': ['1_C06', '3_C12', '6_C05', '6_C08'], '-': ['1_C21', '3_C03', '3_C04', '3_C05', '3_C09', '3_C11'], 'Neutral': ['1_C08', '1_C09', '1_C11', '1_C19', '1_C23', '1_C26', '1_C27', '3_C01', '3_C07', '6_C10']},
}
#RDT 1 Session
rdt_d1 = {
    "mannwhitneyu_Block_Choice_Time_s_1dot0_170_minus10_to_minus5_1_to_4" : {'+': ['1_C13', '1_C18', '3_C01', '3_C10', '6_C02', '6_C14', '11_C02'], '-': ['1_C02', '1_C14', '1_C15', '3_C05', '6_C07', '6_C11', '7_C04', '8_C01', '8_C02'], 'Neutral': ['1_C04', '1_C09', '1_C12', '3_C04', '3_C11', '3_C15', '6_C04', '6_C05', '6_C06', '6_C09', '6_C15', '11_C06', '13_C05', '13_C08', '8_C04', '8_C09', '9_C01', '9_C02', '9_C03', '9_C04', '9_C06', '9_C08']},
"mannwhitneyu_Block_Choice_Time_s_2dot0_170_minus10_to_minus5_1_to_4" : {'+': ['1_C04', '3_C01', '3_C15', '6_C06', '6_C15', '11_C02', '8_C09', '9_C01', '9_C04'], '-': ['1_C14', '1_C15', '3_C04', '3_C05', '3_C10', '6_C07', '6_C11', '13_C08', '8_C01', '8_C02', '9_C02', '9_C06', '9_C08'], 'Neutral': ['1_C02', '1_C09', '1_C12', '1_C13', '1_C18', '3_C11', '6_C02', '6_C04', '6_C05', '6_C09', '6_C14', '7_C04', '11_C06', '13_C05', '8_C04', '9_C03']},
"mannwhitneyu_Block_Choice_Time_s_3dot0_170_minus10_to_minus5_1_to_4" : {'+': ['3_C15', '6_C06', '6_C09', '6_C15', '8_C09'], '-': ['1_C09', '1_C14', '1_C15', '6_C07', '6_C11', '7_C04', '11_C02', '13_C08', '8_C01', '8_C02', '9_C08'], 'Neutral': ['1_C02', '1_C04', '1_C12', '1_C13', '1_C18', '3_C01', '3_C04', '3_C05', '3_C10', '3_C11', '6_C02', '6_C04', '6_C05', '6_C14', '11_C06', '13_C05', '8_C04', '9_C01', '9_C02', '9_C03', '9_C04', '9_C06']}
}

#RDT 2 Session

rdt_d2 = {
    "mannwhitneyu_Block_Choice_Time_s_1dot0_148_minus10_to_minus5_1_to_4" : {'+': ['1_C05', '1_C06', '1_C07', '1_C08', '1_C18', '1_C35', '5_C06', '6_C02', '6_C03', '8_C07', '8_C08', '8_C13'], '-': ['1_C03', '1_C26', '6_C06', '6_C10', '13_C03', '8_C04', '8_C05', '8_C11', '8_C12', '9_C03'], 'Neutral': ['1_C11', '1_C17', '1_C22', '1_C25', '3_C01', '3_C02', '3_C05', '3_C09', '3_C10', '3_C11', '5_C01', '5_C02', '5_C04', '6_C12', '7_C04', '11_C02', '11_C03', '11_C05', '13_C02', '13_C04', '13_C07', '13_C09', '8_C01', '8_C03', '8_C06', '8_C10', '9_C01', '9_C05', '9_C07']},
"mannwhitneyu_Block_Choice_Time_s_2dot0_148_minus10_to_minus5_1_to_4" : {'+': ['1_C05', '1_C06', '1_C08', '1_C18', '1_C35', '3_C09', '3_C10', '3_C11', '5_C04', '8_C06', '8_C11'], '-': ['3_C05', '5_C06', '6_C12', '11_C05', '13_C02', '8_C01', '8_C07', '8_C13'], 'Neutral': ['1_C03', '1_C07', '1_C11', '1_C17', '1_C22', '1_C25', '1_C26', '3_C01', '3_C02', '5_C01', '5_C02', '6_C02', '6_C03', '6_C06', '6_C10', '7_C04', '11_C02', '11_C03', '13_C03', '13_C04', '13_C07', '13_C09', '8_C03', '8_C04', '8_C05', '8_C08', '8_C10', '8_C12', '9_C01', '9_C03', '9_C05', '9_C07']},
"mannwhitneyu_Block_Choice_Time_s_3dot0_148_minus10_to_minus5_1_to_4" : {'+': ['1_C03', '1_C07', '1_C35', '3_C09', '6_C10'], '-': ['1_C05', '1_C18', '1_C22', '3_C05', '3_C10', '5_C06', '6_C06', '13_C02', '13_C03', '8_C01', '8_C04', '8_C12', '8_C13'], 'Neutral': ['1_C06', '1_C08', '1_C11', '1_C17', '1_C25', '1_C26', '3_C01', '3_C02', '3_C11', '5_C01', '5_C02', '5_C04', '6_C02', '6_C03', '6_C12', '7_C04', '11_C02', '11_C03', '11_C05', '13_C04', '13_C07', '13_C09', '8_C03', '8_C05', '8_C06', '8_C07', '8_C08', '8_C10', '8_C11', '9_C01', '9_C03', '9_C05', '9_C07']}
}

pre_rdt_rm_p = perform_chi_square_test([pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_1dot0_57_minus10_to_minus5_1_to_4"], pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_2dot0_57_minus10_to_minus5_1_to_4"], pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_3dot0_57_minus10_to_minus5_1_to_4"]])
rdt_d1_p = perform_chi_square_test([rdt_d1["mannwhitneyu_Block_Choice_Time_s_1dot0_170_minus10_to_minus5_1_to_4"], rdt_d1["mannwhitneyu_Block_Choice_Time_s_2dot0_170_minus10_to_minus5_1_to_4"], rdt_d1["mannwhitneyu_Block_Choice_Time_s_3dot0_170_minus10_to_minus5_1_to_4"]])
rdt_d2_p = perform_chi_square_test([rdt_d2["mannwhitneyu_Block_Choice_Time_s_1dot0_148_minus10_to_minus5_1_to_4"],rdt_d2["mannwhitneyu_Block_Choice_Time_s_2dot0_148_minus10_to_minus5_1_to_4"], rdt_d2["mannwhitneyu_Block_Choice_Time_s_3dot0_148_minus10_to_minus5_1_to_4"]])

print("Pre RDT RM p-value:", pre_rdt_rm_p)
print("RDT 1 p-value:", rdt_d1_p)
print("RDT 2 p-value:", rdt_d2_p)

def compute_expected_frequencies(dicts):
    keys = ['+', '-', 'Neutral']
    total_counts = {key: 0 for key in keys}
    for d in dicts:
        for key in keys:
            total_counts[key] += len(d[key])
    expected = {key: count / len(dicts) for key, count in total_counts.items()}
    return expected

pre_rdt_rm_dicts = [pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_1dot0_57_minus10_to_minus5_1_to_4"], pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_2dot0_57_minus10_to_minus5_1_to_4"], pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_3dot0_57_minus10_to_minus5_1_to_4"]]
expected = compute_expected_frequencies(pre_rdt_rm_dicts + [rdt_d1["mannwhitneyu_Block_Choice_Time_s_1dot0_170_minus10_to_minus5_1_to_4"]])

# Compare across sessions
# Pre RDT RM Session and the first dictionary of RDT 1 Session dictionaries
pre_rdt_rm_and_first_rdt_d1 = [
    pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_1dot0_57_minus10_to_minus5_1_to_4"], pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_2dot0_57_minus10_to_minus5_1_to_4"], pre_rdt_rm["mannwhitneyu_Block_Choice_Time_s_3dot0_57_minus10_to_minus5_1_to_4"],rdt_d1["mannwhitneyu_Block_Choice_Time_s_1dot0_170_minus10_to_minus5_1_to_4"]
]

# RDT D1 Session last two dictionaries
last_two_rdt_d1 = [
    rdt_d1["mannwhitneyu_Block_Choice_Time_s_2dot0_170_minus10_to_minus5_1_to_4"], rdt_d1["mannwhitneyu_Block_Choice_Time_s_3dot0_170_minus10_to_minus5_1_to_4"]
]

# RDT D2 Session dictionaries
rdt_d2 = [
    rdt_d2["mannwhitneyu_Block_Choice_Time_s_1dot0_148_minus10_to_minus5_1_to_4"],rdt_d2["mannwhitneyu_Block_Choice_Time_s_2dot0_148_minus10_to_minus5_1_to_4"], rdt_d2["mannwhitneyu_Block_Choice_Time_s_3dot0_148_minus10_to_minus5_1_to_4"]
]

def get_expected_value(dictionaries):
    total_counts = {}
    for d in dictionaries:
        for k, v in d.items():
            if k not in total_counts:
                total_counts[k] = 0
            total_counts[k] += len(v)
    return {k: v / len(dictionaries) for k, v in total_counts.items()}

expected_value = get_expected_value(pre_rdt_rm_and_first_rdt_d1)

def compare_expected_value_to_dicts(expected_value, dictionaries):
    results = []
    for d in dictionaries:
        #print(d)
        obs = []
        for k, v in d.items():
            print(k, f"observed: {len(v)}", f"expected: {expected_value[k]}")
            #print(k, expected_value[k])
            obs.append([len(v), expected_value[k]])
            chi2, p, _, _ = chi2_contingency(obs)
            results.append((k, p))
            #results.append(p)
    return results

# Compare the expected value to the last two dictionaries of RDT D1 Session
#print(expected_value, last_two_rdt_d1)
rdt_d1_p_values = compare_expected_value_to_dicts(expected_value, last_two_rdt_d1)

# Compare the expected value to all dictionaries of the RDT D2 Session
#print(expected_value, rdt_d2)
rdt_d2_p_values = compare_expected_value_to_dicts(expected_value, rdt_d2)

print("RDT D1 Session last two dictionaries p-values:", rdt_d1_p_values)
print("RDT D2 Session dictionaries p-values:", rdt_d2_p_values)