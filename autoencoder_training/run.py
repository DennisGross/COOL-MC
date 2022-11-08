
import sys
sys.path.insert(0, '../')
from common.tasks.safe_gym_training import *
from common.utilities.project import Project
from common.autoencoders.autoencoder import *
import matplotlib.pyplot as plt
import os
from os import walk

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    command_line_arguments = get_arguments()
    if command_line_arguments['parent_run_id']=="":
        print("No parent run id provided. Exiting.")
        exit(-1)
    command_line_arguments["autoencoder_folder"] = command_line_arguments['prism_file_path'].split(".")[0]+"_data"
    command_line_arguments['deploy'] = 0
    if os.path.exists(command_line_arguments["autoencoder_folder"]) == False:
        os.mkdir(command_line_arguments["autoencoder_folder"])
    set_random_seed(command_line_arguments['seed'])
    
    experiment_id = run_safe_gym_training(command_line_arguments)
    print("Experiment ID:", experiment_id)    
    # Load Project
    env = SafeGym(os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path']), command_line_arguments['constant_definitions'], 
                command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'],
                  command_line_arguments['reward_flag'], 
                  command_line_arguments['seed'], command_line_arguments['permissive_input'],
                  command_line_arguments['disabled_features'])
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'],
        command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments,
                           env.observation_space, env.action_space, all_actions=env.action_mapper.actions)
    m_project.agent.load_env(env)
    print("Project loaded",m_project.command_line_arguments)
    # For which agent do we want to have an autoencoder? -> For all
    autoencoders = []
    # For each autoencoder
    for i in range(len(m_project.agent.agents)):
        # For each state
        m_data_loader = AEDataLoader(command_line_arguments["autoencoder_folder"], m_project.agent, i)

        # Train Autoencoder
        autoencoders.append(AE(m_project.agent.po_manager.get_observation_dimension_for_agent_idx(i)))
        loss_function = torch.nn.MSELoss()
 
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(autoencoders[i].parameters(),
                                    lr = 1e-1,
                                    weight_decay = 1e-8)
        # Plot
        epochs = command_line_arguments['eval_interval']
        outputs = []
        losses = []
        for epoch in range(epochs):
            print("Epoch",epoch)
            for (image, original_image) in m_data_loader:
                # numpy image to tensor
               
                try:
                    image = torch.from_numpy(image).float()
                    image = image.to(device)
                except:
                    try:
                        image = image.to(device)
                    except:
                        pass
                # Output of Autoencoder
                reconstructed = autoencoders[i](image)
                # numpy array to pytorch tensor
                original_image = torch.from_numpy(original_image).float()


                # Original image to pytorch tensor on gpu
                original_image = original_image.to(device)
            
                # Calculating the loss function
                loss = loss_function(reconstructed, original_image)
                
                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Storing the losses in a list for plotting
                losses.append(loss.cpu().detach().numpy())
                outputs.append((epochs, image, reconstructed))
                print("Loss",losses[-1])
        
        # Defining the Plot Style
        #plt.style.use('fivethirtyeight')
        #plt.xlabel('Iterations')
        #plt.ylabel('Loss')
        
        # Plotting the last 100 values
        #plt.plot(losses[-100:])
        #plt.show()
        # Save autoencoder to project copy
        
        
    # Save project copy
    m_project.save()
    m_project.save_autoencoders(autoencoders)