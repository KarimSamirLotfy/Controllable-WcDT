output/tensorboard_logs/20241013-13-39-49_32387 this is the optomized one

output/tensorboard_logs/20241013-13-42-03_97827 this is the normal sampling one


# Comparing the Teacher enforced training

1. diff_run_vis.log (this is the run we are comparing as a baseline)
* 20241013-19-13-49_90670
* 19:13:49 to 22:39:17 == 3hrs 26min
* train/confidence_losses=1.66, train/diffusion_loss=0.99, train/total_loss=3.52, train/traj_losses=0.865


2. opt_diff_run_vis.log (this is the new feature)
* 19:13:17 to 20:39:28 
* 20241013-19-13-17_94936 == 1hr 26 min
* train/confidence_losses=1.65, train/diffusion_loss=1, train/total_loss=3.51, train/traj_losses=0.864 

# Conclusion
* Final loss is very similar, unforutently the tensorboard logs were lost. 
