from experiment import *

## Running


if __name__ == "__main__":
    
    project_name = 'distillation__apdx-fig_13'
    
    temps = np.logspace(-0.5,1.5,20)
    
    for _ in range(5):
        for num_labels in [10]:
            for shuffle in [False,True]:
            
                d = 1000
                for alpha in np.linspace(0.005,0.55,20): 
                    num_samples_per_class = int(alpha * d  * num_labels)
                    
                    data_config = DataConfig(num_samples_per_class=num_samples_per_class,num_labels=num_labels,input_dim=d)
                    X_train, y_train, X_val, y_val = create_dataset(data_config)
                    teacher_config = TeacherConfig(model_name='regression',num_epochs=10000,bias=False)
                    teacher_model, teacher_results = train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val)

                    for train_frac in [0.8]:
                        for temperature in temps:
                            student_config = StudentConfig(train_frac=train_frac, temperature=temperature,  model_name='regression',
                                                           num_epochs=10000,bias=False,shuffle_input_intra_class=shuffle)
                            wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                            student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                            wandb.finish()
                    
