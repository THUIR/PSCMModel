1. user_rating/ : human labeled preference for PSCM model and DBN model, each file represent one anonymous assessor
	format: preference_score(-4,-3,-2,-1:prefer to DBN model, +1,+2,+3,+4: prefer to PSCM model, 0:tie)   query_id
2. query_result_page/ : 
	query_id.txt : each query's Chinese version and the corresponding id 
	css/ : html layout files
	pages/ : each query's search result page, the result list of PSCM model is placed at the left side and the result list of DBN model is placed at the right side (the side information is random on our labeling system)


