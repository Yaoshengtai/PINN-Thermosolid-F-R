from PINN.diff import diff

r1=0.135 #内径
r2=0.1695  #外径
h1=0.02  #高

E= 420*10**9 #杨氏模量
mu =0.14 #泊松比
G=E/2/(1+mu)  #剪切模量
alpha=4*10**-6 #线膨胀系数
beta=alpha *E /(1-2*mu) #热应力系数

def calculate_sigma_rr(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*diff(u,r)/(r2-r1) + mu/(1-2*mu)*(u/((r2-r1)*r+r1)+diff(w,z)/h1)) *10**-12

def calculate_sigma_theta(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*u/((r2-r1)*r+r1) + mu/(1-2*mu)*(diff(w,z)/h1+diff(u,r)/(r2-r1)))*10**-12

def calculate_sigma_zz(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*diff(w,z)/h1 + mu/(1-2*mu)*(u/((r2-r1)*r+r1)+diff(u,r)/(r2-r1)))*10**-12

def calculate_tau_zr(u,w,r,z):
    return G*(diff(w,r)/(r2-r1)+diff(u,z)/h1)*10**-12  ##MPa  um

def force_balance_r(u,w,r,z):
    sigma_rr=calculate_sigma_rr(u,w,r,z)
    sigma_theta=calculate_sigma_theta(u,w,r,z)
    sigma_zz=calculate_sigma_zz(u,w,r,z)
    tau_zr=calculate_tau_zr(u,w,r,z)

    return diff(sigma_rr,r)/(r2-r1)+diff(tau_zr,z)/h1+(sigma_rr-sigma_theta)/((r2-r1)*r+r1)

def force_balance_z(u,w,r,z):
    sigma_rr=calculate_sigma_rr(u,w,r,z)
    sigma_theta=calculate_sigma_theta(u,w,r,z)
    sigma_zz=calculate_sigma_zz(u,w,r,z)
    tau_zr=calculate_tau_zr(u,w,r,z)

    return diff(sigma_zz,z)/h1+diff(tau_zr,r)/(r2-r1)+tau_zr/((r2-r1)*r+r1)