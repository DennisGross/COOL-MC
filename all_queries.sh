#Model Building Time: 8440.37231183052
#Model Size: 11629
#Transitions 27157
#Pmin=? [F ALL_OPS_ZERO=true ]
#Parse Properties...
#Model Checking Time: 0.012749671936035156
#Pmin=? [F ALL_OPS_ZERO=true ] : 0.5862828089454577
#Model Size 11629
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F ALL_OPS_ZERO=true ]"
#Model Building Time: 6432.212338685989
#Model Size: 11479
#Transitions 27007
#Pmin=? [F COLLISION=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.005870819091796875
#Write to file test.drn.
#Pmin=? [F COLLISION=true ] : 0.22329460330932524
#Model Size 11479
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F COLLISION=true ]"
#Model Building Time: 5065.15648317337
#Model Size: 9241
#Transitions 24769
#Pmin=? [F NO_BUDGET_AVAILABLE=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.004967927932739258
#Write to file test.drn.
#Pmin=? [F NO_BUDGET_AVAILABLE=true ] : 0.004158619320844406
#Model Size 9241
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F NO_BUDGET_AVAILABLE=true ]"
#Model Building Time: 8335.766705989838
#Model Size: 11871
#Transitions 27399
#Pmin=? [F TIME_OVER=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.0039365291595458984
#Write to file test.drn.
#Pmin=? [F TIME_OVER=true ] : 4.226941679918599e-21
#Model Size 11871
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F TIME_OVER=true ]"
#Model Building Time: 8162.931264877319
#Model Size: 11711
#Transitions 27239
#Pmin=? [F WRONG_OPERATION_ORDER=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.0036809444427490234
#Write to file test.drn.
#Pmin=? [F WRONG_OPERATION_ORDER=true ] : 0.25163271764904765
#Model Size 11711
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F WRONG_OPERATION_ORDER=true ]"