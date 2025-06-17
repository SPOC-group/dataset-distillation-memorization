from mod_addition_experiment import *

pn = 'distillation__main-fig_2-fig_8'

if __name__ == "__main__":
    
    for _ in range(3):
        
        # run the initial training to get teacher models
        
        init_names = []
        for num_epochs in [2500,5000,10000]:
            config = Config(p=113,frac_train=0.3,lr=0.001, num_epochs=num_epochs,model_type='transformer',num_layers=1, weight_decay=1.0)
            world = Trainer(config = config, project_name=pn)
            init_names.append(wandb.run.name)

            for epoch in range(config.num_epochs):
                    world.train_step()
                    if epoch % config.save_every == 0:
                        world.update_test_metrics()
                        world.save_epoch(epoch)
                        
                        print(f'Epoch {epoch}, train loss {np.log(world.train_losses[-1]):.4f}, test loss {np.log(world.test_losses[-1]):.4f}, train accuracy {world.train_accuracies[-1]:.4f}, test accuracy {world.test_accuracies[-1]:.4f}')
                    
            world.save_epoch(epoch)
            torch.save(world.model.state_dict(), f"{wandb.run.name}-model.pth") 
            wandb.save(f"{wandb.run.name}-model.pth")
            wandb.finish()
            
        
        
        for init_name in init_names:
            for train_frac in [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9]:
                for temp in [0.1,1.0,10.0]:
                    config = Config(p=113,frac_train=0.3,lr=0.001, num_epochs=40000,model_type='transformer',num_layers=1,weight_decay=1.,  transfer_learning_from=init_name,training_label_mode='soft_labels', transfer_learning_temp=temp,transfer_train_fraction=train_frac) 

                    world = Trainer(config = config, project_name=pn)

                    for epoch in range(config.num_epochs):
                            world.train_step()
                            if epoch % config.save_every == 0:
                                world.update_test_metrics()
                                world.save_epoch(epoch)
                                
                                print(f'Epoch {epoch}, train loss {np.log(world.train_losses[-1]):.4f}, test loss {np.log(world.test_losses[-1]):.4f}, train accuracy {world.train_accuracies[-1]:.4f}, test accuracy {world.test_accuracies[-1]:.4f}')
                            
                    world.save_epoch(epoch)
                    torch.save(world.model.state_dict(), f"{wandb.run.name}-model.pth") 
                    wandb.save(f"{wandb.run.name}-model.pth")
                    wandb.finish()
        