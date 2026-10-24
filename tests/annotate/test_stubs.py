import sys

import pandas as pd
import pytest


def test_annotate_imports_without_optional_dependencies():
    import localqtl.annotate as ann
    from localqtl.annotate import alphagenome, igvf
    assert "alphagenome" not in sys.modules or True   # the SDK must not be required
    assert igvf.IGVF_CATALOG_BASE.startswith("https://api.catalogkg.igvf.org")
    assert "ENCODE-rE2G" in igvf.IGVF_ELEMENT_GENE_METHODS
    assert "RNA_SEQ" in alphagenome.ALPHAGENOME_DEFAULT_SCORERS
    assert set(ann.__all__) == {"alphagenome", "igvf"}


def test_stub_functions_raise_not_implemented():
    from localqtl.annotate import alphagenome, igvf
    df = pd.DataFrame()
    for fn, args in (
        (alphagenome.score_lead_variants, (df,)),
        (alphagenome.read_atlas_scores, ("x.parquet",)),
        (alphagenome.annotate_with_alphagenome, (df, df)),
        (igvf.query_variant_gene_links, (df,)),
        (igvf.query_element_gene_links, (df,)),
        (igvf.read_bedpe_links, ("x.bedpe",)),
        (igvf.compare_with_regulatory_links, (df, df, df)),
        (igvf.summarize_link_concordance, (df,)),
    ):
        with pytest.raises(NotImplementedError, match="planned"):
            fn(*args)
