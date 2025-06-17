from logistic_regression_experiment import *
import pickle

if __name__ == "__main__":
    results = []
    d = 200
    for _ in range(5):
        for alpha in np.linspace(0.2,3.5,50):
            num_samples_per_class = int(alpha * d)
            data_config = DataConfig(num_samples_per_class=num_samples_per_class,num_labels=2,input_dim=d)
            X_train, y_train, X_val, y_val = create_dataset(data_config)
            teacher_config = TeacherConfig(model_name='logistic_regression',num_epochs=10000,bias=False)
            teacher_model, teacher_results = train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val)

            for train_frac in np.linspace(0.05, 0.95, 50):
                for shuffle in [True,False]:
                    student_config = StudentConfig(train_frac=train_frac, model_name='logistic_regression',num_epochs=10000,bias=False, shuffle_input_intra_class=shuffle)
                    student_model, student_results = run_student_exp_linear_model(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)

                    ex = {**student_results, **combine_dataclasses(data_config, teacher_config, student_config, teacher_results)}
                    other = {}
                    other['train_acc'] = ex.pop('train_accuracies')[0]
                    other['test_acc'] = ex.pop('test_accuracies')[0]
                    other['test_mse'] = ex.pop('test_mses')[0]
                    other['train_mse'] = ex.pop('train_mses')[0]
                    other['match_teacher_train_acc'] = ex.pop('match_teacher_accs_train')[0]
                    other['match_teacher_test_acc'] = ex.pop('match_teacher_accs_test')[0]
                    other['teacher_train_loss'] = ex.pop('teacher_train_loss')
                    other['teacher_train_acc'] = ex.pop('teacher_train_acc')
                    other['data_num_samples_per_class'] = ex['data_num_samples_per_class']
                    other['student_train_frac'] = ex['student_train_frac']
                    other['student_shuffle_input_intra_class'] = ex['student_shuffle_input_intra_class']
                    other['data_num_labels'] = ex['data_num_labels']
                    del ex 
                    results.append(other)
                    print(results[-1])

    
    with open(f'fig_3_results_linear_with_mse_d={d}.pkl', 'wb') as f:
        pickle.dump(results, f)



