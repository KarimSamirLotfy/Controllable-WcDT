Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1000)'))': /anaconda/pkgs/r/linux-64/repodata.json.zst

## PROPOSED solution by internet
conda config --set ssl_verify false
> Decided not to. as this is the channel without ssl 

https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

And this could lead to expolits with the secruity team. so removeed the anaconda channels and used the default anaconda channels. 
## Stoped the odd channels. and did some changes to the verioining. 
