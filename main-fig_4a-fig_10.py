from experiment import *


if __name__ == "__main__":
    ## example
    project_name = 'distillation__main-fig_4a-fig_10'
    
    temps = np.logspace(-0.5,3.5,40)
    
    for _ in range(5):
        for num_labels in [2]:
            for shuffle in [False,True]:
            
                d = 1000
                for alpha in np.linspace(0.3,1.25,20): 
                    num_samples_per_class = int(alpha * d  * num_labels)
                    data_config = DataConfig(num_samples_per_class=num_samples_per_class,num_labels=num_labels,input_dim=d)
                    X_train, y_train, X_val, y_val = create_dataset(data_config)
                    teacher_config = TeacherConfig(model_name='regression',num_epochs=10000,bias=False)
                    teacher_model, teacher_results = train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val)

                    for train_frac in [0.8]:#np.linspace(0.05, 0.95, 10):
                        for temperature in temps:#[0.5,1.0,5.0,10.,20.0,40.0]:
                            student_config = StudentConfig(train_frac=train_frac, temperature=temperature,  model_name='regression',num_epochs=10000,bias=False,shuffle_input_intra_class=shuffle)
                            wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                            student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                            wandb.finish()
                            
                


