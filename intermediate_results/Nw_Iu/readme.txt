the Iu_ANN_preds.json and Iu_EN_preds.json files were calculated and copied from the project "MetOcean", 
by running the function predict_mean_turbulence_with_ML_at_BJ() with store_predictions=True,
and having mycases[0]['anem_to_test'] == ['bj01', 'bj02', 'bj03', 'bj04', ... , 'bj11']

The MetOcean project only looks at Z=48m. To extrapolate the ANN predictions to other Z, we need to
calculate c0(Z) which depends on the terrain roughness, which in turn is not homogeneous and depends on beta_DB...
To overcome this, we calculate EN predictions at a new Z, e.g. Z=10m. Then use the proportions 
of I_10m_EN_preds and I_48m_EN_preds to find I_10m_ANN_preds! :) This way, the inhomogeneous
terrain roughness is inherently accounted for in the EN procedure (as 2 different roughnesses)
