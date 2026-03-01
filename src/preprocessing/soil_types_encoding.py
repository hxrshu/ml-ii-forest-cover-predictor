import pandas as pd

TEXT_SOIL_TYPES_ENCODING = """1: ELU 2702, Cathedral family - Rock outcrop complex, extremely stony.
2: ELU 2703, Vanet - Ratake families complex, very stony.
3: ELU 2704, Haploborolis - Rock outcrop complex, rubbly.
4: ELU 2705, Ratake family - Rock outcrop complex, rubbly.
5: ELU 2706, Vanet family - Rock outcrop complex complex, rubbly.
6: ELU 2717, Vanet - Wetmore families - Rock outcrop complex, stony.
7: ELU 3501, Gothic family.
8: ELU 3502, Supervisor - Limber families complex.
9: ELU 4201, Troutville family, very stony.
10: ELU 4703, Bullwark - Catamount families - Rock outcrop complex, rubbly.
11: ELU 4704, Bullwark - Catamount families - Rock land complex, rubbly.
12: ELU 4744, Legault family - Rock land complex, stony.
13: ELU 4758, Catamount family - Rock land - Bullwark family complex, rubbly.
14: ELU 5101, Pachic Argiborolis - Aquolis complex.
15: ELU 5151, unspecified in the USFS Soil and ELU Survey.
16: ELU 6101, Cryaquolis - Cryoborolis complex.
17: ELU 6102, Gateview family - Cryaquolis complex.
18: ELU 6731, Rogert family, very stony.
19: ELU 7101, Typic Cryaquolis - Borohemists complex.
20: ELU 7102, Typic Cryaquepts - Typic Cryaquolls complex.
21: ELU 7103, Typic Cryaquolls - Leighcan family, till substratum complex.
22: ELU 7201, Leighcan family, till substratum, extremely bouldery.
23: ELU 7202, Leighcan family, till substratum - Typic Cryaquolls complex.
24: ELU 7700, Leighcan family, extremely stony.
25: ELU 7701, Leighcan family, warm, extremely stony.
26: ELU 7702, Granile - Catamount families complex, very stony.
27: ELU 7709, Leighcan family, warm - Rock outcrop complex, extremely stony.
28: ELU 7710, Leighcan family - Rock outcrop complex, extremely stony.
29: ELU 7745, Como - Legault families complex, extremely stony.
30: ELU 7746, Como family - Rock land - Legault family complex, extremely stony.
31: ELU 7755, Leighcan - Catamount families complex, extremely stony.
32: ELU 7756, Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33: ELU 7757, Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34: ELU 7790, Cryorthents - Rock land complex, extremely stony.
35: ELU 8703, Cryumbrepts - Rock outcrop - Cryaquepts complex.
36: ELU 8707, Bross family - Rock land - Cryumbrepts complex, extremely stony.
37: ELU 8708, Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38: ELU 8771, Leighcan - Moran families - Cryaquolls complex, extremely stony.
39: ELU 8772, Moran family - Cryorthents - Leighcan family complex, extremely stony.
40: ELU 8776, Moran family - Cryorthents - Rock land complex, extremely stony."""


def get_soil_type_mapping(text_encoding=TEXT_SOIL_TYPES_ENCODING):
    mapping = {}
    for line in text_encoding.split("\n"):
        tokens = line.split(
            " "
        )  # Split only on the first space to handle spaces in values
        soil_type_id = int(tokens[0].rstrip(":"))  # Remove the colon and convert to int
        elu_code = tokens[2].rstrip(",")  # Remove the comma and convert to int
        climatic_zone = int(elu_code[0])
        geologic_zone = int(elu_code[1])

        mapping[soil_type_id] = {
            # "elu_code": elu_code,
            "climatic_zone": climatic_zone,
            "geologic_zone": geologic_zone,
        }
    df_mapping = pd.DataFrame.from_dict(mapping, orient="index")
    df_mapping.index.name = "soil_type_id"
    return df_mapping


df_soil_type_mapping = get_soil_type_mapping()


def add_soil_type_breakdown(df_data, soil_type_mapping=df_soil_type_mapping):
    df_raw = df_data.copy()
    df_raw["id"] = df_raw.index
    df_pcd = df_raw.melt(
        id_vars=[col for col in df_raw.columns if not col.startswith("Soil_Type")],
        value_vars=[col for col in df_raw.columns if col.startswith("Soil_Type")],
        var_name="Soil_Type",
        value_name="Value",
    )
    df_pcd = df_pcd[df_pcd["Value"] == 1].drop(columns=["Value"])
    df_pcd["Soil_Type"] = (
        df_pcd["Soil_Type"].str[9:].astype(int)
    )  # Extract the number from "Soil_TypeX" and convert to int
    df_pcd = df_pcd.merge(
        soil_type_mapping, left_on="Soil_Type", right_index=True, how="left"
    )
    # df_pcd = df_pcd.drop(columns=["Soil_Type"])
    df_pcd = pd.get_dummies(
        df_pcd, columns=["Soil_Type", "climatic_zone", "geologic_zone"], dtype=int
    )
    # df_pcd = pd.get_dummies(df_pcd, columns=["climatic_zone", "geologic_zone"], dtype=int)
    df_pcd = df_pcd.set_index("id").reindex(df_raw.index)

    return df_pcd


def remove_soil_type_columns(df):
    df = df.copy()
    soil_type_cols = [col for col in df.columns if col.startswith("Soil_Type")]
    df.drop(columns=soil_type_cols, inplace=True)
    return df
