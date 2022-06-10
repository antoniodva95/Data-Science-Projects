class Judge:
    
    def __init__(self,dataframe_name):
        self.dataframe_name = dataframe_name
        self.X = None
        self.y = None
        self.models = None
        self.algorithm_names = None
        self.params = None
        self.metrics = None
        return None
        
        
    def __str__(self):
        return f" Judging {self.dataframe_name}"
        
    
    def set_data(self,X,y):
        self.X = X
        self.y = y
        return self
    
    def set_algorithms_and_names(self,models,algorithm_names):
        self.models = models
        self.algorithm_names = algorithm_names
        return self
    
    def set_params(self, params):
        self.params = params
        return self
    
    
    def set_metrics(self, metrics):
        self.metrics = metrics
        return self
    
        
    @staticmethod
    def Class_info(cls):
        class_info = f"""Judge class contains methods for optimizing machine learning 
        performance and metrics evaluation for binary classification. It uses nested cross validation.
        
        Methods: 
        set_data -> to introduce data. X for the independent variables, y for the target. 
        set_algorithm_and_names -> to introduce different models (list) and their names (array). 
        set_params -> giving parameters for optimizing performances (dictionary) 
        set_metrics -> to introduce metrics to evaluate
        get_final_performance -> to start the analysis 
        Hyperparameters get_final_performance:
        cv_inner_splits: inner cross validation  
        cv_outer_splits: outer cross validation 
        metric_to_optimize:
        metric to optimize during parameter tuning (default: roc_auc) 
        find_params: set algorithm for parameters search -> GridSearchCV or RandomizedSearchCV(
        default: GridSearchCV)
        Returns:
        Metric performance tab for each model (Dataframe) """
        
        return class_info
    
    
    def __Construct_matrix_df(self,algorithm_names,scores):
    #Costruisco il dataframe a partire 
        Compacted_perf_values = np.hsplit(scores,len(self.metrics)) 
        Performance_matrix = np.append(algorithm_names,Compacted_perf_values).reshape(len(self.metrics)+1,len(self.models)).T 
        columns_vector = np.insert(self.metrics,0,"model") 
        Performance_matrix_df = pd.DataFrame(data=Performance_matrix,columns=columns_vector)
        return Performance_matrix_df
    
    def __get_performance_from_algorithm(self, algorithm, grid, X, y, metrics,inner_cv,outer_cv,find_params,metric_to_optimize):
        
        if grid == {}:
            cvl = cross_validate(algorithm, X, y, scoring = metrics,cv=self.outer_cv)
            results = np.array(list(cvl.values()))[2::,:]
        
        else:
            if self.find_params == "GridSearchCV":
                clf = GridSearchCV(estimator = algorithm, param_grid = grid, scoring = metrics,refit=metric_to_optimize,cv=self.inner_cv)
            elif self.find_params == "RandomizedSearchCV":
                clf = RandomizedSearchCV(estimator = algorithm, param_distributions= grid, scoring = metrics,refit=metric_to_optimize,cv=self.inner_cv)
            cvl = cross_validate(clf, X, y, scoring = metrics,cv=self.outer_cv)
            results = np.array(list(cvl.values()))[2::,:]
        
        results = np.mean(results,axis=1)
        return np.round(results*100,2)
    
    def get_final_performance(self,cv_inner_splits,cv_outer_splits,metric_to_optimize = "roc_auc",find_params="GridSearchCV"):
        self.inner_cv = KFold(n_splits=cv_inner_splits,shuffle=True)
        self.outer_cv = KFold(n_splits=cv_outer_splits,shuffle=True)
        self.find_params = find_params
        self.metric_to_optimize = metric_to_optimize
        self.scores = np.zeros((len(self.models),len(self.metrics))) #Initializing scoring matrix
        for model in range(len(self.models)):
            grid = {}
            if self.algorithm_names[model] in self.params.keys(): #Sets grid for parameters
                grid = self.params[self.algorithm_names[model]]
            
            score = self.__get_performance_from_algorithm(self.models[model],grid,self.X,self.y,self.metrics,self.inner_cv,self.outer_cv,self.find_params,self.metric_to_optimize) 
            self.scores[model] = score 
        
        
        Performance_matrix_df = self.__Construct_matrix_df(self.algorithm_names,self.scores)
        return Performance_matrix_df