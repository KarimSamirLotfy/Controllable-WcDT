Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1000)'))': /anaconda/pkgs/r/linux-64/repodata.json.zst

## PROPOSED solution by internet
conda config --set ssl_verify false
> Decided not to. as this is the channel without ssl 

https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

And this could lead to expolits with the secruity team. so removeed the anaconda channels and used the default anaconda channels. 
## Stoped the odd channels. and did some changes to the verioining. 


# Runiing for baseline runs. but It seesm GPU is underutalized and cpu is over utalized. 


# This repo does not save trained models. aka we must do so. Very weird

# TODO
[x] Must train quicker. it is too slow. not using GPU enough. I think it is not even using batching process. 
[x] this repo also can save gifs. which could be more usefull 
[x] has show results that creates gifs. use it to create some gifs for visualisations
[x] add tensorboard
[x] results

[] Analyse decisions made by team
[] look at how to actually add query centric 
[] Think about how will we do conditinoing/ samplgin control. 

# SHOW the results of the teacher forcing


[] Do with full data
[ ] get the eval code. 
[] decidec on to keep the arch or not. 
# also ask for more space. Run ID: 

# Time needed now
* 50000 samples, 512 batch size, 2 GPUs, Teacher_forced=True. 100 epochs => 13hrs
    * 98 batches
    * 1 batch = 4.7 seconds on 2 GPUS
    * 1 batch = 9.4 seconds on 1 GPU
    * num_batches * 9.4 * num_epochs / num_gpu = seconds == num_hrs*60*60

* 50000, 256 batch size, 1 gpu, Teacher_forced=True, 200 epochs => 
    * 196
    * 1 batch = 3 seconds on 1 GPU
    
f(batch_szie) = time_per_second
f(256) = 3s
f(512) = 9.4
m = 0.026
b = -3.656
time_per_seconds = 0.026*batch_size

# Meeting ponts
* Metrics issue. 
* without quanititative results. it is hard to judge. 
* 10x Dataset definelty improved losses. except of the confidence loss. 
* Maybe mix the teacher force with none. get best of both words

* Next step
    1. Start adding query centric
    2. or go for the sampling based techniques
        * so for the paper. try to see if the agent tokens really do change the behavior. 
        * My idea. as you sample each step, you woudl forward propagate the entire thing accting as if the entire thing is just 1 MLP that maps from beahvior to output. then prepagate the output back
        * Dissucuss what would be a good result to have. this will not need any trianing. but will need a lot of enginnering to get right. still need also to look at energy based models

# TODO Tasks
[x] Do quick run comapring diffusion model and non diffuison model. to actually see how important this is. 
[x] Fix metrics (Fixed by creating new dataloader that is used during evaluation)
[]  Add sampling condigitning 

* Get the sze of the dataset
* Number of scenarios in the dataset: 486995
* Ratio = 50000/486995 = 0.1 = 10% So we are using 10 percent of the data
 ## Min effort
 * increate number of prediceted
 * edit scenario.... somehow. hackit 
 * implement the metrics. 


# ANOTHER MODEL is SMART
* Won the 2024 sim agents. but it is autoregressive.
* Fact that autoregressive always wins means that, Maybe seeing the prediction of each agent helps the model
* Codebase could be used to create diffusion model with state of the art. by stopping autoregressive and actually going for a more diffusion based arch
* This one would make sence as it uses latent diffuison. so adversrial would just be finding the motion token that causes the most harm. 