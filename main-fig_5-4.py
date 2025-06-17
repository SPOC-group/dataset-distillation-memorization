from experiment import *


if __name__ == "__main__":

    project_name = 'distillation__main-fig_5'

    for _ in range(5):
        for c in [20,50,90,110]:
            for num_samples_per_class in [50, 125, 250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1375, 1500, 1625, 1750, 1875, 2000, 2125, 2250, 2375, 2500]:
                data_config = DataConfig(num_samples_per_class=num_samples_per_class,num_labels=c,input_dim=1000)
                X_train, y_train, X_val, y_val = create_dataset(data_config)
                teacher_config = TeacherConfig(model_name='one_layer_network',hidden_dim=500,num_epochs=2000)
                teacher_model, teacher_results = train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val)

                for train_frac in [0.45]:
                    for temperature in [20.0]:
                        student_config = StudentConfig(train_frac=train_frac, temperature=temperature,  model_name='one_layer_network',hidden_dim=500,num_epochs=10000)
                        wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                        student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                        wandb.finish()


