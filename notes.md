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