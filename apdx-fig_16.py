from experiment import *


if __name__ == "__main__":

    project_name = 'distillation__apdx-fig_16'
    hidden_dim = 500
    num_epochs = 10000
    
    
    for _ in range(5):

        for num_samples_per_class in [750,1250,1500,1750,250,500,1000,2000]:
            data_config = DataConfig(num_samples_per_class=num_samples_per_class,num_labels=100,input_dim=1000)
            X_train, y_train, X_val, y_val = create_dataset(data_config)
            teacher_config = TeacherConfig(model_name='one_layer_network',hidden_dim=500,num_epochs=1000)
            teacher_model, teacher_results = train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val)

            for train_frac in [0.7]:
                for temperature in [20.0]:
                    
                    student_config = StudentConfig(train_frac=train_frac, temperature=temperature, hidden_dim=hidden_dim, model_name='one_layer_network',num_epochs=num_epochs,
                                                   shuffle_input_intra_class=True)
                    wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                    student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                    wandb.finish()
                    
                    student_config = StudentConfig(train_frac=train_frac, temperature=temperature, hidden_dim=hidden_dim, model_name='one_layer_network',num_epochs=num_epochs,)
                    wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                    student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                    wandb.finish()
                    
                    for softmax_treatment in [('cut_tail', 0.02),('remove_logits', [0]),('cut_tail', 0.01), ('cut_tail', 0.1), ('cut_tail', 0.5)]:
                            student_config = StudentConfig(train_frac=train_frac, temperature=temperature, hidden_dim=hidden_dim, soft_label_treatment=softmax_treatment, model_name='one_layer_network',num_epochs=num_epochs)
                            wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                            student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                            wandb.finish()
                    
                    for softmax_filtering in [('remove_labels', [0])]:
                            student_config = StudentConfig(train_frac=train_frac, temperature=temperature, hidden_dim=hidden_dim, soft_label_filtering=softmax_filtering, model_name='one_layer_network',num_epochs=num_epochs,)
                            wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                            student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                            wandb.finish()
                    


