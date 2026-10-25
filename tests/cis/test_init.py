import src.localqtl.cis as cis


def test_cis_init_exports_symbols():
    expected = {"map_nominal", "map_nominal_loci", "map_permutations", "map_independent",
                "map_conditional_dosage", "CisMapper"}
    assert set(cis.__all__) == expected

    # Spot-check that the exported callables are usable
    assert callable(cis.map_nominal)
    assert callable(cis.map_permutations)
    assert callable(cis.map_independent)
    assert callable(cis.map_conditional_dosage)
    assert isinstance(cis.CisMapper.__name__, str)
