from typing import List, Tuple
from protac_splitter.protac_splitter import split_protac
import pandas as pd

def protac_examples() -> List[Tuple[str, str]]:
    return [
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]CCCCCN2CCN(c1ccc(C([*:1])=O)cc1)CC2.[*:2]Nc3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CN(c1ccc(C#N)c(Cl)c1)[C@H]1CC[C@H](NC(=O)c2ccc(N3CC(CN4CCN(c5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)C3)cc2)CC1',
            '[*:1]N[C@@H]2CC[C@@H](N(C)c1ccc(C#N)c(Cl)c1)CC2.[*:1]C(=O)c3ccc(N2CC(CN1CCN([*:2])CC1)C2)cc3.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CN1C(=O)CCc2cc3cc(c21)OCCOCC1CN(C(=O)CCC(=O)NCCCOCCOCCOc2cccc4c2C(=O)N(C2CCC(=O)NC2=O)C4=O)CCN1c1ncc(Cl)c(n1)N3',
            '[*:1]N5CCN4c1ncc(Cl)c(n1)Nc3cc2CCC(=O)N(C)c2c(c3)OCCOCC4C5.[*:2]OCCOCCOCCCNC(=O)CCC([*:1])=O.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'N#CC1(CNc2cccc(-c3cc(N[C@H]4CC[C@H](NCC(=O)NCCOCCOCCOCCNc5cccc6c5C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)ncc3Cl)n2)CCOCC1',
            '[*:1]C(=O)CN[N@@H]4CC[C@@H](Nc3cc(c2cccc(NCC1(C#N)CCOCC1)n2)c(Cl)cn3)CC4.[*:2]NCCOCCOCCOCCN[*:1].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCOCCN4CCN(Cc5ccc6nc(NC(=O)c7cccc(C(F)(F)F)c7)n([C@H]7CC[C@@H](CO)CC7)c6c5)CC4)cc3C2=O)C(=O)N1',
            '[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3.[*:1]CN1CCN(CCOCCOCCOCCO[*:2])CC1.[*:1]c4ccc3nc(NC(=O)c1cccc(C(F)(F)F)c1)n([C@@H]2CC[C@H](CO)CC2)c3c4',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCNc5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCCN2CCN(c1ccc(C([*:1])=O)cc1)CC2.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)COCCOCCOCCNC(=O)CCC(=O)N2CCN([C@H]3CC[C@@H](Nc4ncnn5ccc(C(C)C)c45)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]C(=O)CCC(=O)NCCOCCOCCOCC([*:2])=O.[*:1]N4CCN([C@@H]3CC[C@H](Nc1ncnn2ccc(C(C)C)c12)CC3)CC4',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CNC(=O)c2cccc(-c3ccc(N4CCN(C)CC4)c(NC(=O)c4c[nH]c(=O)cc4C(F)(F)F)c3)c2)C(C)(C)C)cc1',
            '[*:2]NC(C(=O)N1CC(O)CC1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]NCC([*:2])=O.[*:1]C(=O)c4cccc(c3ccc(N1CCN(C)CC1)c(NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)c3)c4',
        ],
        [
            'CN1C(=O)CCc2cc3cc(c21)OCCOC[C@H]1CN(C(=O)CCC(=O)NCCCOCCOCCOc2cccc4c2C(=O)N(C2CCC(=O)NC2=O)C4=O)CCN1c1ncc(Cl)c(n1)N3',
            '[*:1]N5CCN4c1ncc(Cl)c(n1)Nc3cc2CCC(=O)N(C)c2c(c3)OCCOC[C@H]4C5.[*:2]OCCOCCOCCCNC(=O)CCC([*:1])=O.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCOCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)[C@@H]4CC[C@@H](NC(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C)CC4.[*:1]NCCOCCOCCOCCNC(=O)CO[*:2].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],

        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)[C@@H]4CC[C@@H](NC(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C)CC4.[*:1]NCCOCCOCCNC(=O)CO[*:2].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CCCNC(=O)CCC(=O)N2CCN([C@H]3CC[C@@H](Nc4ncnn5ccc(C(C)C)c45)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:2]C(=O)CCCNC(=O)CCC([*:1])=O.[*:1]N4CCN([C@@H]3CC[C@H](Nc1ncnn2ccc(C(C)C)c12)CC3)CC4',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C.[*:1]N[C@@H]1CC[C@@H](C(=O)NCCCCNC(=O)CO[*:2])CC1.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CN2CCN(CCN3CCC(O[C@H]4C[C@H](Oc5ccc6c(c5)Sc5cc([N+](=O)[O-])ccc5N6)C4)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)N[C@@H](C)c3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]O[C@@H]3C[C@@H](OC2CCN(CCN1CCN(CC([*:2])=O)CC1)CC2)C3.[*:1]c3ccc2[nH]c1ccc(N(=O)=O)cc1sc2c3',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCCCCCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCCCCCCCNc1ccc(C([*:1])=O)cc1.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCCCCCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]CCC[*:1].[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CC(=O)N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)N[C@@H](CC(=O)N1CCC(N2CCC(C#Cc3ccc(C(=O)N[C@H]4C(C)(C)[C@H](Oc5ccc(C#N)c(Cl)c5)C4(C)C)cc3)CC2)CC1)c1ccc(-c2scnc2C)cc1)C(C)(C)C',
            'CC(=O)N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)[*:2])C(C)(C)C.Cc1ncsc1-c1ccc([C@H](CC(=O)N2CCC(N3CCC(C#C[*:1])CC3)CC2)N[*:2])cc1.CC1(C)[C@H](NC(=O)c2ccc([*:1])cc2)C(C)(C)[C@H]1Oc1ccc(C#N)c(Cl)c1',
        ],
    ]


def test_split_protac_io_formats():
    """Test split_protac input/output formats for string, list, and DataFrame inputs."""
    examples = protac_examples()
    # Use a small subset for format testing
    protac_str = examples[0][0]
    protac_list = [ex[0] for ex in examples[:3]]
    df = pd.DataFrame({'text': protac_list})

    # Test string input
    result_str = split_protac(protac_str, fix_predictions=True, use_transformer=False, use_xgboost=True)
    assert isinstance(result_str, dict), "String input should return a dict."
    assert 'default_pred_n0' in result_str, "Output dict missing 'default_pred_n0'."
    assert 'model_name' in result_str, "Output dict missing 'model_name'."
    assert isinstance(result_str['default_pred_n0'], (str, type(None))), "Prediction should be a string or None."

    # Test list input
    result_list = split_protac(protac_list, fix_predictions=True, use_transformer=False, use_xgboost=True)
    assert isinstance(result_list, list), "List input should return a list."
    assert all(isinstance(r, dict) for r in result_list), "Each item in list output should be a dict."
    assert all('default_pred_n0' in r for r in result_list), "Each dict missing 'default_pred_n0'."
    assert all(isinstance(r['default_pred_n0'], (str, type(None))) for r in result_list), "Each prediction should be a string or None."

    # Test DataFrame input
    result_df = split_protac(df, fix_predictions=True, use_transformer=False, use_xgboost=True)
    assert isinstance(result_df, pd.DataFrame), "DataFrame input should return a DataFrame."
    assert 'default_pred_n0' in result_df.columns, "Output DataFrame missing 'default_pred_n0' column."
    assert 'model_name' in result_df.columns, "Output DataFrame missing 'model_name' column."
    assert result_df['default_pred_n0'].apply(lambda x: isinstance(x, (str, type(None)))).all(), "Each prediction should be a string or None."


def test_split_protac_io_combinations():
    """Test split_protac I/O formats for various model, batch_size, and num_proc combinations."""
    examples = protac_examples()
    protac_str = examples[0][0]
    protac_list = [ex[0] for ex in examples[:3]]
    df = pd.DataFrame({'text': protac_list})

    model_combos = [
        {'use_transformer': True, 'use_xgboost': True},
        {'use_transformer': False, 'use_xgboost': False},
        {'use_transformer': False, 'use_xgboost': True},
        {'use_transformer': True, 'use_xgboost': False},
    ]
    batch_sizes = [1, 2]
    num_procs = [1, 2]

    for model_args in model_combos:
        for batch_size in batch_sizes:
            for num_proc in num_procs:
                # String input
                result_str = split_protac(
                    protac_str,
                    fix_predictions=True,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    **model_args
                )
                assert isinstance(result_str, dict), f"String input should return a dict for args {model_args}, batch_size={batch_size}, num_proc={num_proc}."
                assert 'default_pred_n0' in result_str
                assert 'model_name' in result_str
                assert isinstance(result_str['default_pred_n0'], (str, type(None)))

                # List input
                result_list = split_protac(
                    protac_list,
                    fix_predictions=True,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    **model_args
                )
                assert isinstance(result_list, list), f"List input should return a list for args {model_args}, batch_size={batch_size}, num_proc={num_proc}."
                assert all(isinstance(r, dict) for r in result_list)
                assert all('default_pred_n0' in r for r in result_list)
                assert all(isinstance(r['default_pred_n0'], (str, type(None))) for r in result_list)

                # DataFrame input
                result_df = split_protac(
                    df,
                    fix_predictions=True,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    **model_args
                )
                assert isinstance(result_df, pd.DataFrame), f"DataFrame input should return a DataFrame for args {model_args}, batch_size={batch_size}, num_proc={num_proc}."
                assert 'default_pred_n0' in result_df.columns
                assert 'model_name' in result_df.columns
                assert result_df['default_pred_n0'].apply(lambda x: isinstance(x, (str, type(None)))).all()