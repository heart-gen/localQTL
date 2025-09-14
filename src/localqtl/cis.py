class SimpleCisMapper:
    def __init__(self, genotype_reader, phenotype_df, phenotype_pos_df,
                 covariates_df=None, haplotypes=None, device="auto"):
        # genotype_reader: like tensorQTL genotypeio.InputGeneratorCis
        self.geno_reader = genotype_reader
        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df
        self.device = "cuda" if device=="auto" and torch.cuda.is_available() else "cpu"

        # residualize phenotypes
        if covariates_df is not None:
            self.residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(self.device))
            self.Y = self._residualize(phenotype_df.values, covariates_df.values)
        else:
            self.residualizer = None
            self.Y = phenotype_df.values

        # handle haplotypes (drop one ancestry col)
        if haplotypes is not None:
            self.H = haplotypes[:, :, :-1]  # (variants × samples × n_pops-1)
        else:
            self.H = None

    def map_nominal(self, window=1_000_000):
        """Nominal cis-QTL scan"""
        results = []
        for pheno_id, pheno_vec, geno_block, var_df in self.geno_reader.generate_data(window=window):
            pheno_t = torch.tensor(pheno_vec, dtype=torch.float32, device=self.device)
            geno_t = torch.tensor(geno_block, dtype=torch.float32, device=self.device)
            impute_mean(geno_t)

            if self.H is not None:
                hap_t = torch.tensor(self.H[var_df.index.values], device=self.device)
                # concatenate genotype+haplotype as predictors
                X = torch.cat([geno_t.unsqueeze(1), hap_t], dim=1)
            else:
                X = geno_t.unsqueeze(1)

            # fast correlation method
            tstat, slope, se = self._regress(X, pheno_t)
            df = pd.DataFrame({
                "variant_id": var_df.index,
                "phenotype_id": pheno_id,
                "tstat": tstat.cpu().numpy(),
                "slope": slope.cpu().numpy(),
                "slope_se": se.cpu().numpy(),
            })
            results.append(df)
        return pd.concat(results)

    def map_permutations(self, nperm=1000, window=1_000_000):
        """Permutation-based empirical cis-QTLs"""
        # same loop, but shuffle phenotype each time and record max r2
        pass

    def map_independent(self, cis_df, fdr=0.05):
        """Forward–backward conditional mapping"""
        pass

    # --- helper functions ---
    def _residualize(self, Y, C):
        C_t = torch.tensor(C, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        CtC_inv = torch.linalg.inv(C_t.T @ C_t)
        proj = C_t @ (CtC_inv @ (C_t.T @ Y_t))
        return (Y_t - proj).cpu().numpy()

    def _regress(self, X, y):
        # closed-form OLS
        XtX = X.T @ X
        XtX_inv = torch.linalg.inv(XtX)
        betas = XtX_inv @ (X.T @ y)
        y_hat = X @ betas
        resid = y - y_hat
        dof = X.shape[0] - X.shape[1]
        sigma2 = (resid @ resid) / dof
        se = torch.sqrt(torch.diag(XtX_inv) * sigma2)
        tstats = betas / se
        return tstats, betas, se
