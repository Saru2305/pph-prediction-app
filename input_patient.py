def get_patient_data():

    print("\nEnter Patient Details\n")

    patient_data = {}

    patient_data['age'] = int(input("1 Age: "))
    patient_data['bmi'] = float(input("2 BMI: "))
    patient_data['hemoglobin_level'] = float(input("3 Hemoglobin Level: "))
    patient_data['anemia'] = int(input("4 Anemia (1=yes,0=no): "))
    patient_data['pre_pph'] = int(input("5 Previous PPH (1=yes,0=no): "))
    patient_data['parity'] = int(input("6 Parity: "))
    patient_data['multiple_pregnancy'] = int(input("7 Multiple Pregnancy (1=yes,0=no): "))
    patient_data['pre_hypertension'] = int(input("8 Pre Hypertension (1=yes,0=no): "))
    patient_data['placenta_previa'] = int(input("9 Placenta Previa (1=yes,0=no): "))
    patient_data['gest_diabetes'] = int(input("10 Gestational Diabetes (1=yes,0=no): "))
    patient_data['pre_c_section'] = int(input("11 Previous C Section (1=yes,0=no): "))
    patient_data['pre_blood'] = int(input("12 Blood Disorder (1=yes,0=no): "))
    patient_data['pre_aipabnormal_placenta'] = int(input("13 Abnormal Placenta (1=yes,0=no): "))
    patient_data['polyhydromnios'] = int(input("14 Polyhydramnios (1=yes,0=no): "))
    patient_data['hellp_syndrome'] = int(input("15 HELLP Syndrome (1=yes,0=no): "))
    patient_data['severe_preeclampsia'] = int(input("16 Severe Preeclampsia (1=yes,0=no): "))
    patient_data['surgery'] = int(input("17 Surgery (1=yes,0=no): "))
    patient_data['myoma'] = int(input("18 Myoma (1=yes,0=no): "))
    patient_data['gestational_age'] = int(input("19 Gestational Age (weeks): "))

    return patient_data
