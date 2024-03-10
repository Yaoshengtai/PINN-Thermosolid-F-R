from PINN.diff import diff

r1=0.135 #内径
r2=0.1695  #外径
h1=0.02  #高

E= 420*10**-3 #杨氏模量 um $ MPa
mu =0.14 #泊松比
G=E/2/(1+mu)  #剪切模量
alpha=4*10**-6 #线膨胀系数
beta=alpha *E /(1-2*mu) #热应力系数

def calculate_sigma_rr(u,w,r,z):
    #return 2*G * ((1-mu)/(1-2*mu)*diff(u,r) + mu/(1-2*mu)*(u/r+diff(w,z)))
    return 2*G * ((1-mu)/(1-2*mu)*diff(u,r) + mu/(1-2*mu)*(u/r+diff(w,z)))

def calculate_sigma_theta(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*u/r + mu/(1-2*mu)*(diff(w,z)+diff(u,r)))

def calculate_sigma_zz(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*diff(w,z) + mu/(1-2*mu)*((u/r)+diff(u,r)))

def calculate_tau_zr(u,w,r,z):
    return G*(diff(w,r)+diff(u,z))  ##MPa  um

def force_balance_r(u,w,r,z):
    sigma_rr=calculate_sigma_rr(u,w,r,z)
    sigma_theta=calculate_sigma_theta(u,w,r,z)
    tau_zr=calculate_tau_zr(u,w,r,z)

    return diff(sigma_rr,r)+diff(tau_zr,z)+(sigma_rr-sigma_theta)/r

def force_balance_z(u,w,r,z):
    sigma_zz=calculate_sigma_zz(u,w,r,z)
    tau_zr=calculate_tau_zr(u,w,r,z)

    return diff(sigma_zz,z)+diff(tau_zr,r)+tau_zr/r